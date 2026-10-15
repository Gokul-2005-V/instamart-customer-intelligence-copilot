"""Run the seven SQL analysis queries and verify reference result checkpoints."""
import json
import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    sql_text = (ROOT / 'sql' / 'queries.sql').read_text(encoding='utf-8')
    queries = [q.strip() for q in sql_text.split(';') if q.strip()]
    results = {}
    with sqlite3.connect(ROOT / 'data' / 'instamart.db') as conn:
        frames = [pd.read_sql_query(q, conn) for q in queries]
    q1, q2, q3, q4, q5, q6, q7 = frames
    with sqlite3.connect(ROOT / 'data' / 'instamart.db') as conn:
        average_all_customers = pd.read_sql_query("SELECT AVG(total_spend) AS avg_spend FROM (SELECT customer_id, SUM(order_amount) AS total_spend FROM orders WHERE status='completed' GROUP BY customer_id)", conn).iloc[0, 0]
    results['Q1'] = {'rows': len(q1)}
    results['Q2'] = {'rows': len(q2), 'total_revenue_inr': round(float(q2.revenue_inr.sum()), 2), 'largest_cell': q2.sort_values('revenue_inr', ascending=False).iloc[0].to_dict()}
    results['Q3'] = {'rows': len(q3)}
    results['Q4'] = {'rows': len(q4)}
    results['Q5'] = {'rows': len(q5), 'average_customer_spend_inr': round(float(average_all_customers), 2)}
    results['Q6'] = {'rows': len(q6), 'rank_differs_from_row_number': int((q6.spend_rank != q6.spend_row_number).sum()), 'top_customer_by_city': {city: {'customer_id': row.customer_id, 'total_spend_inr': round(float(row.total_spend), 2)} for city, row in q6.sort_values(['city', 'spend_rank', 'customer_id']).groupby('city').first().iterrows()}}
    results['Q7'] = {'rows': len(q7), 'last_cumulative_revenue_inr': round(float(q7.cumulative_revenue_inr.iloc[-1]), 2)}
    checks = {
        'Q1 customer count': results['Q1']['rows'] == 30,
        'Q2 row count': results['Q2']['rows'] == 30,
        'Q2 revenue total': results['Q2']['total_revenue_inr'] == 2998668.80,
        'Q2 largest city-category': results['Q2']['largest_cell']['city'] == 'Bengaluru' and results['Q2']['largest_cell']['category'] == 'Electronics' and round(results['Q2']['largest_cell']['revenue_inr'], 2) == 233909.45,
        'Q3 product count': results['Q3']['rows'] == 11,
        'Q4 customer count': results['Q4']['rows'] == 25,
        'Q5 above-average count': results['Q5']['rows'] == 550,
        'Q5 mean across qualifying customers': results['Q5']['average_customer_spend_inr'] == 2552.06,
        'Q6 rank ties': results['Q6']['rank_differs_from_row_number'] == 8,
        'Q7 days': results['Q7']['rows'] == 183,
        'Q7 cumulative revenue': results['Q7']['last_cumulative_revenue_inr'] == 2998668.80,
    }
    results['checks'] = checks
    (ROOT / 'outputs' / 'part_a_sql_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    for name, value in results.items():
        print(f'{name}: {value}')
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise AssertionError('SQL checkpoint failures: ' + ', '.join(failed))
    print('ALL SQL CHECKPOINTS PASSED')

if __name__ == '__main__':
    main()
