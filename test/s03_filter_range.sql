-- S03: Range filters (BETWEEN, <, >, <=, >=)
SELECT l_orderkey, l_linenumber, l_extendedprice
FROM lineitem
WHERE l_orderkey >= 100 AND l_orderkey < 200
  AND l_extendedprice BETWEEN 1000 AND 5000
ORDER BY l_orderkey, l_linenumber
