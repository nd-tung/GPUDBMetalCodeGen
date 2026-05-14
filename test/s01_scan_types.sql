-- S01: Scan different column types from lineitem
SELECT l_orderkey, l_linenumber, l_extendedprice, l_discount, l_returnflag, l_linestatus, l_shipdate, l_shipmode, l_comment
FROM lineitem
WHERE l_orderkey BETWEEN 1 AND 5
ORDER BY l_orderkey, l_linenumber
