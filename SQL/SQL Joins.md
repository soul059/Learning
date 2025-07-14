---
tags:
  - sql
  - joins
  - relational-algebra
  - combining-tables
---

# SQL Joins

[[SQL|SQL Joins]] are used to combine rows from two or more tables based on a related column between them. This is a fundamental operation in [[Relational Model|relational databases]] for querying data that is spread across multiple tables. Joins are a practical implementation of concepts derived from [[Relational Algebra]] operations like Cartesian Product ($\times$) and Selection ($\sigma$), often combined with Projection ($\Pi$).

The most common syntax uses the `JOIN` keyword within the `FROM` clause of a [[SQL DQL|SELECT]] statement.

## Types of SQL Joins

Let's consider two tables, `TableA` and `TableB`.

```sql
-- Example Tables for illustration
CREATE TABLE TableA (
    AID INT PRIMARY KEY,
    NameA VARCHAR(50)
);

CREATE TABLE TableB (
    BID INT PRIMARY KEY,
    AID INT, -- Foreign Key referencing TableA
    NameB VARCHAR(50)
);

INSERT INTO TableA VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie');
INSERT INTO TableB VALUES (101, 1, 'Apple'), (102, 1, 'Avocado'), (103, 2, 'Banana'), (104, 4, 'Blueberry'); -- Note: AID 4 in TableB has no match in TableA
```

### 1. INNER JOIN

* **Purpose:** Returns only the rows where the join condition is met in **both** tables. Non-matching rows are excluded.
* **Analogy:** Similar to the [[Relational Algebra|Relational Algebra]] Natural Join ($\natural$), but requires an explicit `ON` condition (unless columns have the same name, which is less common with `INNER JOIN`).
* **Keyword:** `INNER JOIN` (or just `JOIN`, as `INNER` is the default).

```sql
SELECT A.NameA, B.NameB
FROM TableA A
INNER JOIN TableB B ON A.AID = B.AID;
```

* **Result for Example Data:**
    | NameA | NameB   |
    |-------|---------|
    | Alice | Apple   |
    | Alice | Avocado |
    | Bob   | Banana  |

### 2. LEFT JOIN (or LEFT OUTER JOIN)

* **Purpose:** Returns all rows from the **left** table (`TableA` in the syntax above), and the matched rows from the right table (`TableB`). If there is no match in the right table, [[SQL NULL Values|NULL]] values are returned for columns from the right table.
* **Analogy:** Corresponds to the [[Relational Algebra|Relational Algebra]] Left Outer Join ($\enhancedisplaysci{\mathrel{\text{\textbf{⟕}}}}$).
* **Keyword:** `LEFT JOIN` or `LEFT OUTER JOIN`.

```sql
SELECT A.NameA, B.NameB
FROM TableA A
LEFT JOIN TableB B ON A.AID = B.AID;
```

* **Result for Example Data:** (Charlie from TableA has no match in TableB)
    | NameA   | NameB   |
    |---------|---------|
    | Alice   | Apple   |
    | Alice   | Avocado |
    | Bob     | Banana  |
    | Charlie | NULL    |

### 3. RIGHT JOIN (or RIGHT OUTER JOIN)

* **Purpose:** Returns all rows from the **right** table (`TableB` in the syntax above), and the matched rows from the left table (`TableA`). If there is no match in the left table, [[SQL NULL Values|NULL]] values are returned for columns from the left table.
* **Analogy:** Corresponds to the [[Relational Algebra|Relational Algebra]] Right Outer Join ($\enhancedisplaysci{\mathrel{\text{\textbf{⟖}}}}$).
* **Keyword:** `RIGHT JOIN` or `RIGHT OUTER JOIN`.

```sql
SELECT A.NameA, B.NameB
FROM TableA A
RIGHT JOIN TableB B ON A.AID = B.AID;
```

* **Result for Example Data:** (Blueberry from TableB has no match in TableA)
    | NameA | NameB     |
    |-------|-----------|
    | Alice | Apple     |
    | Alice | Avocado   |
    | Bob   | Banana    |
    | NULL  | Blueberry |

### 4. FULL JOIN (or FULL OUTER JOIN)

* **Purpose:** Returns all rows when there is a match in either the left or the right table. If there is no match, [[SQL NULL Values|NULL]] values are returned for the columns from the table that doesn't have a match. It is the combination of LEFT and RIGHT joins.
* **Analogy:** Corresponds to the [[Relational Algebra|Relational Algebra]] Full Outer Join ($\enhancedisplaysci{\mathrel{\text{\textbf{⟗}}}}$).
* **Keyword:** `FULL JOIN` or `FULL OUTER JOIN`. *Not supported in MySQL versions prior to 8.0; can be simulated using `LEFT JOIN` + `UNION` + `RIGHT JOIN`.*

```sql
SELECT A.NameA, B.NameB
FROM TableA A
FULL JOIN TableB B ON A.AID = B.AID;
```

* **Result for Example Data:** (Includes Charlie from A and Blueberry from B)
    | NameA   | NameB     |
    |---------|-----------|
    | Alice   | Apple     |
    | Alice   | Avocado   |
    | Bob     | Banana    |
    | Charlie | NULL      |
    | NULL    | Blueberry |

### 5. CROSS JOIN

* **Purpose:** Returns the [[Relational Algebra|Cartesian Product]] of the rows from the joined tables. It combines each row from the first table with each row from the second table. There is no `ON` clause for `CROSS JOIN`.
* **Analogy:** Corresponds directly to the [[Relational Algebra|Relational Algebra]] Cartesian Product ($\times$).
* **Keyword:** `CROSS JOIN`.

```sql
SELECT A.NameA, B.NameB
FROM TableA A
CROSS JOIN TableB B;
```

* **Result for Example Data:** (3 rows in A * 4 rows in B = 12 rows)
    | NameA   | NameB     |
    |---------|-----------|
    | Alice   | Apple     |
    | Alice   | Avocado   |
    | Alice   | Banana    |
    | Alice   | Blueberry |
    | Bob     | Apple     |
    | Bob     | Avocado   |
    | Bob     | Banana    |
    | Bob     | Blueberry |
    | Charlie | Apple     |
    | Charlie | Avocado   |
    | Charlie | Banana    |
    | Charlie | Blueberry |

Choosing the correct join type depends entirely on whether you need to include non-matching rows from one or both tables.

---
**Related Notes:**
* [[SQL]]
* [[SQL DQL]]
* [[Relational Model]]
* [[Relational Algebra]]
* [[SQL NULL Values]]