-- MB1: minimal column scan + sum (no filter)
-- Isolates: GridStrideScan + TGReduce with the smallest column footprint
-- (1B char col). Useful as a near-zero-bandwidth baseline.

SELECT
    SUM(l_returnflag) AS s
FROM
    lineitem;
