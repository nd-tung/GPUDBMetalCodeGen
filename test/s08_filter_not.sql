-- S08: NOT filter
SELECT l_orderkey, l_linenumber
FROM lineitem
WHERE NOT (l_linenumber = 1)
  AND l_orderkey < 10
ORDER BY l_orderkey, l_linenumber
