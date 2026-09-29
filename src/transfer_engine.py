import pandas as pd
import numpy as np

def generate_transfer_recommendations(df_preds):
    # df_preds contains: store_id, product_id, current_stock, incoming_stock, recommended_stock, shortage, surplus, risk_level, stockout_probability
    
    recommendations = []
    
    # We only process products that have at least one shortage
    products_with_shortage = df_preds[df_preds['shortage'] > 0]['product_id'].unique()
    
    # Group by product to do transfers
    for prod in products_with_shortage:
        prod_data = df_preds[df_preds['product_id'] == prod]
        
        # Sources: stores with surplus > 0
        sources = prod_data[prod_data['surplus'] > 0].sort_values(
            by=['surplus', 'stockout_probability'], ascending=[False, True]
        ).to_dict('records')
        
        # Destinations: stores with shortage > 0
        destinations = prod_data[prod_data['shortage'] > 0].sort_values(
            by=['stockout_probability', 'shortage'], ascending=[False, False]
        ).to_dict('records')
        
        for dest in destinations:
            remaining_shortage = dest['shortage']
            dest_id = dest['store_id']
            transfers = []
            
            for src in sources:
                if remaining_shortage <= 0:
                    break
                if src['surplus'] <= 0:
                    continue
                if src['store_id'] == dest_id:
                    continue
                    
                transfer_qty = min(remaining_shortage, src['surplus'])
                remaining_shortage -= transfer_qty
                src['surplus'] -= transfer_qty
                
                transfers.append({
                    'source_store_id': src['store_id'],
                    'transfer_quantity': transfer_qty
                })
            
            # Finalize for this destination
            if transfers:
                if len(transfers) == 1:
                    action = "TRANSFER + PLACE SUPPLIER ORDER" if remaining_shortage > 0 else "TRANSFER FROM STORE"
                else:
                    action = "TRANSFER FROM MULTIPLE STORES + PLACE SUPPLIER ORDER" if remaining_shortage > 0 else "TRANSFER FROM MULTIPLE STORES"
            else:
                action = "PLACE SUPPLIER ORDER"
                
            rec = {
                'date': str(dest.get('date', '')),
                'store_id': dest_id,
                'product_id': prod,
                'category': dest.get('category', ''),
                'sub_category': dest.get('sub_category', ''),
                'brand': dest.get('brand', ''),
                'city': dest.get('city', ''),
                'store_type': dest.get('store_type', ''),
                'region': dest.get('region', ''),
                'supplier_id': dest.get('supplier_id', ''),
                'cost_price': dest.get('cost_price', 0.0),
                'avg_selling_price': dest.get('avg_selling_price', 0.0),
                'days_of_inventory': dest.get('days_of_inventory', 0.0),
                'reorder_gap': dest.get('reorder_gap', 0.0),
                'promotion_flag': dest.get('promotion_flag', 0),
                'weekend_flag': dest.get('weekend_flag', 0),
                'demand_growth_7': dest.get('demand_growth_7', 1.0),
                'temp_c': dest.get('temp_c', 25.0),
                'rain_mm': dest.get('rain_mm', 0.0),
                'current_stock': dest['current_stock'],
                'incoming_stock': dest['incoming_stock'],
                'forecast_7d_demand': dest['forecast_7d_demand'],
                'safety_stock': dest['safety_stock'],
                'recommended_stock': dest['recommended_stock'],
                'stockout_probability': dest['stockout_probability'],
                'risk_level': str(dest['risk_level']),
                'shortage_before_transfer': dest['shortage'],
                'source_store_id': ", ".join([str(t['source_store_id']) for t in transfers]) if transfers else "N/A",
                'transfer_quantity': sum(t['transfer_quantity'] for t in transfers),
                'remaining_shortage': remaining_shortage,
                'supplier_order_quantity': remaining_shortage, # Supplier order is the remaining shortage
                'action': action,
                'reason': f"Shortage of {dest['shortage']:.1f}. Covered {sum(t['transfer_quantity'] for t in transfers):.1f} by transfer.",
                'source_surplus_before_transfer': sum(t['transfer_quantity'] for t in transfers), # Rough proxy
                'source_remaining_stock_after_transfer': 'N/A'
            }
            recommendations.append(rec)
            
        # Add those without shortage but maybe need monitoring
        no_shortage = prod_data[prod_data['shortage'] <= 0].to_dict('records')
        for ns in no_shortage:
            recommendations.append({
                'date': str(ns.get('date', '')),
                'store_id': ns['store_id'],
                'product_id': prod,
                'category': ns.get('category', ''),
                'sub_category': ns.get('sub_category', ''),
                'brand': ns.get('brand', ''),
                'city': ns.get('city', ''),
                'store_type': ns.get('store_type', ''),
                'region': ns.get('region', ''),
                'supplier_id': ns.get('supplier_id', ''),
                'cost_price': ns.get('cost_price', 0.0),
                'avg_selling_price': ns.get('avg_selling_price', 0.0),
                'days_of_inventory': ns.get('days_of_inventory', 0.0),
                'reorder_gap': ns.get('reorder_gap', 0.0),
                'promotion_flag': ns.get('promotion_flag', 0),
                'weekend_flag': ns.get('weekend_flag', 0),
                'demand_growth_7': ns.get('demand_growth_7', 1.0),
                'temp_c': ns.get('temp_c', 25.0),
                'rain_mm': ns.get('rain_mm', 0.0),
                'current_stock': ns['current_stock'],
                'incoming_stock': ns['incoming_stock'],
                'forecast_7d_demand': ns['forecast_7d_demand'],
                'safety_stock': ns['safety_stock'],
                'recommended_stock': ns['recommended_stock'],
                'stockout_probability': ns['stockout_probability'],
                'risk_level': str(ns['risk_level']),
                'shortage_before_transfer': 0,
                'source_store_id': "N/A",
                'transfer_quantity': 0,
                'remaining_shortage': 0,
                'supplier_order_quantity': 0,
                'action': "MONITOR INVENTORY" if str(ns['risk_level']) != 'LOW' else "NO ACTION REQUIRED",
                'reason': "Sufficient inventory",
                'source_surplus_before_transfer': ns['surplus'],
                'source_remaining_stock_after_transfer': ns['surplus']
            })
            
    # Also add products with no shortages anywhere
    products_no_shortage = set(df_preds['product_id'].unique()) - set(products_with_shortage)
    for prod in products_no_shortage:
        prod_data = df_preds[df_preds['product_id'] == prod].to_dict('records')
        for ns in prod_data:
            recommendations.append({
                'date': str(ns.get('date', '')),
                'store_id': ns['store_id'],
                'product_id': prod,
                'category': ns.get('category', ''),
                'sub_category': ns.get('sub_category', ''),
                'brand': ns.get('brand', ''),
                'city': ns.get('city', ''),
                'store_type': ns.get('store_type', ''),
                'region': ns.get('region', ''),
                'supplier_id': ns.get('supplier_id', ''),
                'cost_price': ns.get('cost_price', 0.0),
                'avg_selling_price': ns.get('avg_selling_price', 0.0),
                'days_of_inventory': ns.get('days_of_inventory', 0.0),
                'reorder_gap': ns.get('reorder_gap', 0.0),
                'promotion_flag': ns.get('promotion_flag', 0),
                'weekend_flag': ns.get('weekend_flag', 0),
                'demand_growth_7': ns.get('demand_growth_7', 1.0),
                'temp_c': ns.get('temp_c', 25.0),
                'rain_mm': ns.get('rain_mm', 0.0),
                'current_stock': ns['current_stock'],
                'incoming_stock': ns['incoming_stock'],
                'forecast_7d_demand': ns['forecast_7d_demand'],
                'safety_stock': ns['safety_stock'],
                'recommended_stock': ns['recommended_stock'],
                'stockout_probability': ns['stockout_probability'],
                'risk_level': str(ns['risk_level']),
                'shortage_before_transfer': 0,
                'source_store_id': "N/A",
                'transfer_quantity': 0,
                'remaining_shortage': 0,
                'supplier_order_quantity': 0,
                'action': "MONITOR INVENTORY" if str(ns['risk_level']) != 'LOW' else "NO ACTION REQUIRED",
                'reason': "Sufficient inventory",
                'source_surplus_before_transfer': ns['surplus'],
                'source_remaining_stock_after_transfer': ns['surplus']
            })

    return pd.DataFrame(recommendations)
