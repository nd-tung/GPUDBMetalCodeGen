-- S21: Simple 2-table join
SELECT o_orderkey, o_orderdate, l_quantity
FROM lineitem, orders
WHERE l_orderkey = o_orderkey
  AND o_orderkey BETWEEN 1 AND 10
ORDER BY o_orderkey
