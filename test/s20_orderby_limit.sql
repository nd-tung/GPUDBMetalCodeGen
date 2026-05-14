-- S20: ORDER BY + LIMIT (int column, descending)
SELECT l_orderkey, l_extendedprice, l_discount
FROM lineitem
WHERE l_orderkey BETWEEN 1 AND 100
ORDER BY l_orderkey DESC
LIMIT 10
