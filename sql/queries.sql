-- Q1: Successful recharge revenue by plan.
SELECT
    p.plan_name,
    SUM(r.amount_inr) AS total_revenue_inr,
    ROUND(AVG(r.amount_inr), 2) AS average_recharge_inr,
    COUNT(DISTINCT r.subscriber_id) AS distinct_subscribers
FROM recharges AS r
INNER JOIN subscribers AS s
    ON s.subscriber_id = r.subscriber_id
INNER JOIN plans AS p
    ON p.plan_id = s.plan_id
WHERE r.status = 'success'
GROUP BY p.plan_name
ORDER BY total_revenue_inr DESC;

-- Q2: Subscribers with no successful recharge.
SELECT
    COUNT(DISTINCT s.subscriber_id) AS never_successful_count,
    COUNT(DISTINCT CASE
        WHEN s.tenure_months = 0 THEN s.subscriber_id
    END) AS zero_tenure_count
FROM subscribers AS s
LEFT JOIN recharges AS r
    ON r.subscriber_id = s.subscriber_id
    AND r.status = 'success'
WHERE r.recharge_id IS NULL;

-- Q3: Successful recharge spend and three ranking functions within each city.
WITH subscriber_spend AS (
    SELECT
        s.subscriber_id,
        s.city,
        COALESCE(SUM(r.amount_inr), 0) AS total_successful_recharge_inr
    FROM subscribers AS s
    LEFT JOIN recharges AS r
        ON r.subscriber_id = s.subscriber_id
        AND r.status = 'success'
    GROUP BY s.subscriber_id, s.city
)
SELECT
    subscriber_id,
    city,
    total_successful_recharge_inr,
    RANK() OVER (
        PARTITION BY city
        ORDER BY total_successful_recharge_inr DESC
    ) AS spend_rank,
    DENSE_RANK() OVER (
        PARTITION BY city
        ORDER BY total_successful_recharge_inr DESC
    ) AS dense_spend_rank,
    ROW_NUMBER() OVER (
        PARTITION BY city
        ORDER BY total_successful_recharge_inr DESC, subscriber_id
    ) AS spend_row_number
FROM subscriber_spend
ORDER BY city, spend_row_number;

-- Q4: Successful recharges and running total for TP00010.
SELECT
    recharge_id,
    subscriber_id,
    recharge_date,
    amount_inr,
    SUM(amount_inr) OVER (
        ORDER BY recharge_date, recharge_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total_inr
FROM recharges
WHERE subscriber_id = 'TP00010'
    AND status = 'success'
ORDER BY recharge_date, recharge_id;
