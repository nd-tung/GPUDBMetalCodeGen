-- S30: COUNT(DISTINCT) with GROUP BY
SELECT l_linenumber, COUNT(DISTINCT l_orderkey) AS uniq_orders
FROM lineitem
WHERE l_orderkey < 1000
GROUP BY l_linenumber
ORDER BY l_linenumber
