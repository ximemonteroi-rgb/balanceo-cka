import os
from flask import Flask, request, jsonify
import pandas as pd
import gspread
from google.oauth2.credentials import Credentials
import pulp

app = Flask(__name__)

# Función de seguridad para validar el token que envía Sheets
def get_credentials_from_header(auth_header):
    if not auth_header or not auth_header.startswith('Bearer '):
        return None
    token = auth_header.split(' ')[1]
    return Credentials(token)

@app.route('/')
def home():
    return "Servicio de Nivelación de Holdings API Funcionando"

@app.route('/optimizar', methods=['POST'])
def ejecutar_nivelacion():
    # 1. Autenticación con el token del usuario (recibido desde Sheets)
    auth_header = request.headers.get('Authorization')
    creds = get_credentials_from_header(auth_header)
    
    if not creds:
        return jsonify({"error": "No se proporcionó un token de autorización válido"}), 401

    try:
        # 2. Conectar a Google Sheets
        gc = gspread.authorize(creds)
        spreadsheet_url = 'https://docs.google.com/spreadsheets/d/1ZGMq5g2kuQlDFlngXCC281xGFD_ZGudQbL-WS-u5s5Y/edit'
        sh = gc.open_by_url(spreadsheet_url)
        worksheet_portfolio = sh.worksheet('portfolio')
        worksheet_summary = sh.worksheet('Summary')

        # 3. Leer Inputs
        escenario = worksheet_summary.acell('B10').value
        bd_objetivo = worksheet_summary.acell('C10').value

        current_bds_raw = worksheet_summary.get('B15:B23')
        bd_list_base = [row[0].strip() for row in current_bds_raw if row and str(row[0]).strip() not in ('', '0')]

        data_summary = worksheet_summary.get('B32:G')
        df_shops = pd.DataFrame(data_summary[1:], columns=[
            'Brand_name_corregido', 'BD', 'holding_id', 'Shops', 'Avg Daily Orders', 'Puede reasignarse esta brand?'
        ])

        # 4. Limpieza de datos
        df_shops['Avg Daily Orders'] = pd.to_numeric(df_shops['Avg Daily Orders'], errors='coerce').fillna(0)
        df_shops['Shops'] = pd.to_numeric(df_shops['Shops'], errors='coerce').fillna(0)
        df_shops['is_fixed'] = df_shops['Puede reasignarse esta brand?'].str.strip().str.lower() == 'no'
        df_shops['brands_in_holding'] = df_shops.groupby('holding_id')['Brand_name_corregido'].transform('nunique')

        # 5. Lógica de Escenarios
        bd_list = bd_list_base.copy()
        if escenario == 'Termination' and bd_objetivo:
            bd_objetivo = bd_objetivo.strip()
            if bd_objetivo in bd_list:
                bd_list.remove(bd_objetivo)
            cuentas_liberadas = df_shops['BD'] == bd_objetivo
            df_shops.loc[cuentas_liberadas, 'is_fixed'] = False
        elif escenario == 'New Hire' and bd_objetivo:
            bd_objetivo = bd_objetivo.strip()
            if bd_objetivo not in bd_list:
                bd_list.append(bd_objetivo)

        # 6. Agrupación por Holding
        df_holdings = df_shops.groupby('holding_id').agg(
            total_do=('Avg Daily Orders', 'sum'),
            total_shops=('Shops', 'sum'),
            total_brands=('Brand_name_corregido', 'nunique'),
            is_reassignable=('is_fixed', lambda val: not val.any()),
            current_bd=('BD', lambda val: val.mode()[0] if not val.mode().empty else None)
        ).reset_index()

      # 7. Modelo de Optimización (Sintaxis Blindada para Render)
        holdings = df_holdings['holding_id'].tolist()
        bds = bd_list

        h_do = df_holdings.set_index('holding_id')['total_do'].to_dict()
        h_shops = df_holdings.set_index('holding_id')['total_shops'].to_dict()
        h_brands = df_holdings.set_index('holding_id')['total_brands'].to_dict() # <-- AÑADIDO: Diccionario de marcas
        h_reassign = df_holdings.set_index('holding_id')['is_reassignable'].to_dict()
        h_current_bd = df_holdings.set_index('holding_id')['current_bd'].to_dict()

        prob = pulp.LpProblem("Balanceo_Holdings", pulp.LpMinimize)

        # Usando la sintaxis manual que funcionó en Render
        x = {}
        for h in holdings:
            x[h] = {}
            for b in bds:
                nombre = f"Asig_{str(h).replace(' ', '_').replace('.', '_')}_{str(b).replace(' ', '_').replace('.', '_')}"
                x[h][b] = pulp.LpVariable(nombre, 0, 1, pulp.LpBinary)

        min_do = pulp.LpVariable("Min_DO", 0, None, pulp.LpContinuous)
        max_do = pulp.LpVariable("Max_DO", 0, None, pulp.LpContinuous)
        min_sh = pulp.LpVariable("Min_Sh", 0, None, pulp.LpContinuous)
        max_sh = pulp.LpVariable("Max_Sh", 0, None, pulp.LpContinuous)
        min_br = pulp.LpVariable("Min_Br", 0, None, pulp.LpContinuous)
        max_br = pulp.LpVariable("Max_Br", 0, None, pulp.LpContinuous)

        # Pesos Dinámicos
        sum_do = sum(h_do.values())
        sum_sh = sum(h_shops.values())
        sum_br = sum(h_brands.values())

        peso_do = 1
        peso_sh = (sum_do / sum_sh) if sum_sh > 0 else 1
        peso_br = (sum_do / sum_br) if sum_br > 0 else 1

        prob += peso_do * (max_do - min_do) + peso_sh * (max_sh - min_sh) + peso_br * (max_br - min_br)

        for h in holdings:
            prob += pulp.lpSum(x[h][b] for b in bds) == 1
            if not h_reassign[h]:
                bd_fijo = h_current_bd[h]
                if bd_fijo in bds:
                    prob += x[h][bd_fijo] == 1

        for b in bds:
            c_do = pulp.lpSum(x[h][b] * h_do[h] for h in holdings)
            c_sh = pulp.lpSum(x[h][b] * h_shops[h] for h in holdings)
            c_br = pulp.lpSum(x[h][b] * h_brands[h] for h in holdings) # <-- AÑADIDO: Cálculo de marcas

            prob += c_do >= min_do
            prob += c_do <= max_do
            prob += c_sh >= min_sh
            prob += c_sh <= max_sh
            prob += c_br >= min_br # <-- AÑADIDO: Restricción de marcas
            prob += c_br <= max_br # <-- AÑADIDO: Restricción de marcas

        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=120))
        
        status_modelo = pulp.LpStatus[prob.status]

        if prob.status == 1:
            # 8. Actualizar Sheets si es óptimo
            asig_map = {h: b for h in holdings for b in bds if pulp.value(x[h][b]) > 0.5}
            df_shops['Assigned_BD'] = df_shops['holding_id'].map(asig_map)
            df_shops['Hubo_Cambio'] = df_shops.apply(
                lambda row: "SÍ" if str(row['BD']).strip() != str(row['Assigned_BD']).strip() else "NO", axis=1
            )

            dict_assigned = df_shops.set_index('Brand_name_corregido')['Assigned_BD'].to_dict()
            dict_cambio = df_shops.set_index('Brand_name_corregido')['Hubo_Cambio'].to_dict()

            portfolio_data = worksheet_portfolio.get_all_values()
            headers = portfolio_data[0]

            try:
                idx_brand = headers.index('Brand_name_corregido')
            except ValueError:
                idx_brand = 2

            updates_al_am = [["", ""] for _ in range(len(portfolio_data))]
            updates_al_am[0] = ["Propuesta 1", "Hubo_cambios"]

            for row_idx, row in enumerate(portfolio_data[1:], start=1):
                brand_name = str(row[idx_brand]).strip()
                if brand_name in dict_assigned:
                    updates_al_am[row_idx] = [dict_assigned[brand_name], dict_cambio[brand_name]]

            rango_update = f'AL1:AM{len(updates_al_am)}'
            worksheet_portfolio.update(values=updates_al_am, range_name=rango_update)
            
            return jsonify({"mensaje": f"Nivelación Exitosa. Estado: {status_modelo}"}), 200
        else:
            return jsonify({"error": "El optimizador no pudo encontrar una solución óptima en el tiempo límite."}), 500

    except Exception as e:
        import traceback
        error_detallado = traceback.format_exc()
        print("ERROR FATAL:\n", error_detallado)
        return jsonify({"error": error_detallado}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
