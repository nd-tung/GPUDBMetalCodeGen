-- S05: LIKE filter on CHAR_FIXED column
SELECT l_orderkey, l_linenumber, l_shipmode, l_comment
FROM lineitem
WHERE l_shipmode LIKE 'MA%'
  AND l_orderkey < 100
ORDER BY l_orderkey, l_linenumber
LIMIT 10
