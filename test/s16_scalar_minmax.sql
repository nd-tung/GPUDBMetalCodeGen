-- S16: Scalar MIN/MAX
SELECT MIN(l_quantity) AS min_qty,
       MAX(l_quantity) AS max_qty,
       MIN(l_extendedprice) AS min_price,
       MAX(l_extendedprice) AS max_price
FROM lineitem
WHERE l_orderkey < 100
