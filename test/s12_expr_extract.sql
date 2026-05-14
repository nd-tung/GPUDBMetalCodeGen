-- S12: Date extraction (year = col / 10000)
SELECT l_orderkey, l_shipdate, l_shipdate / 10000 AS ship_year
FROM lineitem
WHERE l_orderkey < 50
  AND l_shipdate >= 19920101
ORDER BY l_orderkey, l_linenumber
LIMIT 10
