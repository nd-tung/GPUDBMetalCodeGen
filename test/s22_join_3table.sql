-- S22: 3-table chain join
SELECT c_name, o_orderkey, l_quantity
FROM lineitem, orders, customer
WHERE l_orderkey = o_orderkey
  AND o_custkey = c_custkey
  AND o_orderkey BETWEEN 1 AND 10
ORDER BY o_orderkey
