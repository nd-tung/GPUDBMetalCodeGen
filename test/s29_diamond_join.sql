-- S29: Diamond join pattern (nation shared by customer and supplier)
SELECT c_name, s_name, n_name
FROM customer, orders, lineitem, supplier, nation
WHERE c_custkey = o_custkey
  AND o_orderkey = l_orderkey
  AND l_suppkey = s_suppkey
  AND c_nationkey = n_nationkey
  AND s_nationkey = n_nationkey
  AND o_orderkey BETWEEN 1 AND 100
LIMIT 5
