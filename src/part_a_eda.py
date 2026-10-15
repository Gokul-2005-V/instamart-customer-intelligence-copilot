"""Part A: pandas cross-checks and descriptive statistics (no charts committed)."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    orders = pd.read_csv(ROOT / 'data' / 'clean' / 'orders.csv', parse_dates=['order_ts'])
    customers = pd.read_csv(ROOT / 'data' / 'clean' / 'customers.csv')
    products = pd.read_csv(ROOT / 'data' / 'clean' / 'products.csv')
    completed = orders.loc[orders.status.eq('completed')].copy()
    completed = completed.merge(customers[['customer_id', 'city']], on='customer_id', validate='many_to_one')
    completed = completed.merge(products[['product_id', 'category']], on='product_id', validate='many_to_one')

    pivot = pd.pivot_table(completed, values='order_amount', index='city', columns='category', aggfunc='sum', fill_value=0, margins=True, margins_name='All')
    city = completed.groupby('city', as_index=False).agg(
        sum_inr=('order_amount', 'sum'), mean_inr=('order_amount', 'mean'), count=('order_amount', 'count')
    )
    amount = completed['order_amount'].astype(float)
    q1, median, q3 = amount.quantile([0.25, 0.5, 0.75]).tolist()
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    correlations = completed[['order_amount', 'discount_pct', 'quantity', 'delivery_minutes']].corr()
    pairs = []
    cols = list(correlations.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            pairs.append({'variable_1': a, 'variable_2': b, 'correlation': float(correlations.loc[a, b]), 'absolute_correlation': abs(float(correlations.loc[a, b]))})
    strongest = sorted(pairs, key=lambda x: x['absolute_correlation'], reverse=True)[0]

    result = {
        'pivot_table_with_margins': {str(k): {str(c): round(float(v), 2) for c, v in row.items()} for k, row in pivot.round(2).iterrows()},
        'pivot_grand_total_inr': round(float(pivot.loc['All', 'All']), 2),
        'city_summary': [{**row, 'sum_inr': round(float(row['sum_inr']), 2), 'mean_inr': round(float(row['mean_inr']), 2), 'count': int(row['count'])} for row in city.to_dict('records')],
        'completed_order_amount': {
            'count': int(amount.count()), 'mean_inr': round(float(amount.mean()), 2), 'median_inr': round(float(median), 2), 'skewness': round(float(amount.skew()), 2),
            'q1_inr': round(float(q1), 2), 'q3_inr': round(float(q3), 2), 'iqr_inr': round(float(iqr), 2),
            'lower_fence_inr': round(float(lower), 2), 'upper_fence_inr': round(float(upper), 2),
            'outliers_below_lower_fence': int((amount < lower).sum()), 'outliers_above_upper_fence': int((amount > upper).sum()),
        },
        'correlation_matrix': {str(k): {str(c): round(float(v), 6) for c, v in row.items()} for k, row in correlations.round(6).iterrows()},
        'strongest_absolute_correlation_pair': strongest,
        'interpretation_under_200_words': (
            'The dataset describes fictional customer orders across six cities and product categories. Completed order amounts are strongly right-skewed: '
            f'the mean is INR {amount.mean():.2f}, above the INR {median:.2f} median, with skewness {amount.skew():.2f}. '
            f'The 1.5×IQR upper fence is INR {upper:.2f}; {(amount > upper).sum()} orders exceed it and {(amount < lower).sum()} fall below the lower fence. '
            f'The largest absolute pairwise correlation is {strongest["variable_1"]} versus {strongest["variable_2"]} '
            f'({strongest["correlation"]:.3f}). This correlation is small and does not establish causation.'
        ),
    }
    assert result['pivot_grand_total_inr'] == 2998668.80
    assert result['completed_order_amount']['outliers_above_upper_fence'] == 897
    assert result['completed_order_amount']['outliers_below_lower_fence'] == 0
    (ROOT / 'outputs' / 'part_a_eda.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    print('ALL EDA CHECKPOINTS PASSED')

if __name__ == '__main__':
    main()
