---
tags:
  - sql
  - dql
  - select
  - query
  - sql-clauses
---

# SQL DQL (Data Query Language - SELECT)

[[SQL|DQL (Data Query Language)]] is primarily concerned with retrieving data from the database. The core command is the `SELECT` statement. It is the most frequently used SQL command and has numerous clauses to specify exactly what data to retrieve and how to present it.

The basic structure of a `SELECT` statement is:

```sql
SELECT column1, column2, ...  -- What columns to retrieve
FROM table1                   -- From which table(s)
[JOIN table2 ON join_condition] -- How to combine tables (if needed)
[WHERE condition]             -- Which rows to filter
[GROUP BY column(s)]          -- How to group rows
[HAVING group_condition]      -- Which groups to filter
[ORDER BY column(s) [ASC|DESC]] -- How to sort the results
[LIMIT number | OFFSET number] -- How many rows to return (syntax varies)
```
*(Clauses in `[]` are optional)*

Let's break down the key clauses:

## SELECT Clause

Specifies the columns you want to see in the result set.

* `SELECT *`: Selects all columns from the table(s).
* `SELECT column1, column2, ...`: Selects only the specified columns.
* `SELECT DISTINCT column1, ...`: Selects unique values from the specified columns. Duplicates are removed.
* `SELECT column1 AS alias_name, ...`: Renames a column in the result set using an alias.
* `SELECT expression`: Can include arithmetic expressions or function calls (like [[SQL Aggregate Functions and Grouping|aggregate functions]]).

**Examples:**

```sql
SELECT * FROM Students; -- Get all columns and all rows from Students
```

```sql
SELECT FirstName, LastName, MajorID FROM Students; -- Get specific columns
```

```sql
SELECT DISTINCT MajorID FROM Students; -- Get unique Major IDs
```

```sql
SELECT StudentID, FirstName, LastName AS FamilyName FROM Students; -- Rename LastName to FamilyName
```

```sql
SELECT StudentID, FirstName, (2025 - YEAR(DateOfBirth)) AS Age FROM Students; -- Calculate Age (example assuming DateOfBirth exists)
```

## FROM Clause

Specifies the table(s) from which to retrieve data. Multiple tables can be listed, typically combined using [[SQL Joins|JOIN]] operations.

```sql
FROM table_name;
```
```sql
FROM table1 JOIN table2 ON table1.column = table2.column; -- Using JOIN
```

**Example:**

```sql
SELECT * FROM Students;
```
```sql
SELECT * FROM Students, Majors; -- Cartesian Product (usually avoided directly)
```

## JOIN Clause

Used to combine rows from two or more tables based on a related column between them. See [[SQL Joins]] for detailed types and examples.

```sql
SELECT S.FirstName, S.LastName, M.MajorName
FROM Students S -- Using table aliases
JOIN Majors M ON S.MajorID = M.MajorID;
```

## WHERE Clause

Filters rows based on a specified condition. It is applied *before* any grouping.

* **Condition:** Uses comparison operators (`=, !=, %3C, %3E, <=, >=`), logical operators (`AND`, `OR`, `NOT`), `LIKE` (pattern matching), `IN` (list of values), `BETWEEN` (range), [[SQL NULL Values|IS NULL]], [[SQL NULL Values|IS NOT NULL]].

**Examples:**

```sql
SELECT * FROM Students WHERE MajorID = 1;
```

```sql
SELECT * FROM Students WHERE EnrollmentDate >= '2024-01-01' AND MajorID IS NOT NULL;
```

```sql
SELECT * FROM Students WHERE LastName LIKE 'Sm%'; -- Last names starting with 'Sm'
```

```sql
SELECT * FROM Students WHERE StudentID IN (101, 105, 110);
```

## GROUP BY Clause

Groups rows that have the same values in specified columns into summary rows, like "find the number of students in each major". It is typically used with [[SQL Aggregate Functions and Grouping|aggregate functions]].

```sql
GROUP BY column1, column2, ...;
```

**Example:**

```sql
SELECT MajorID, COUNT(StudentID) AS NumberOfStudents
FROM Students
GROUP BY MajorID; -- Count students per major
```

## HAVING Clause

Filters groups created by the `GROUP BY` clause based on a group condition. It is applied *after* grouping and aggregation. Conditions often involve [[SQL Aggregate Functions and Grouping|aggregate functions]].

```sql
HAVING condition;
```

**Example:**

```sql
-- Find majors that have more than 5 students
SELECT MajorID, COUNT(StudentID) AS NumberOfStudents
FROM Students
GROUP BY MajorID
HAVING COUNT(StudentID) > 5; -- Filter groups where count is > 5
```
*Note: `WHERE` filters rows *before* grouping. `HAVING` filters groups *after* grouping.*

## ORDER BY Clause

Sorts the result set based on one or more columns.

* `ASC`: Ascending order (default).
* `DESC`: Descending order.

```sql
ORDER BY column1 [ASC|DESC], column2 [ASC|DESC], ...;
```

**Example:**

```sql
-- Order students by last name, then first name, ascending
SELECT FirstName, LastName
FROM Students
ORDER BY LastName ASC, FirstName ASC;
```

```sql
-- Order students by enrollment date, newest first
SELECT FirstName, LastName, EnrollmentDate
FROM Students
ORDER BY EnrollmentDate DESC;
```

## LIMIT / TOP Clause

Restricts the number of rows returned by the query. Syntax varies by RDBMS.

* `LIMIT number`: Standard SQL (PostgreSQL, MySQL, SQLite). Returns the first `number` rows.
* `LIMIT number OFFSET offset`: Returns `number` rows after skipping `offset` rows.
* `TOP number`: SQL Server. Returns the top `number` rows.
* `ROWNUM <= number`: Oracle (used in `WHERE` clause).

**Examples:**

```sql
-- Get the top 10 students by StudentID (example using LIMIT)
SELECT * FROM Students ORDER BY StudentID LIMIT 10;
```

```sql
-- Get the next 10 students after the first 10 (for pagination, using LIMIT and OFFSET)
SELECT * FROM Students ORDER BY StudentID LIMIT 10 OFFSET 10;
```

```sql
-- Get the top 5 students by CPI (example using TOP, SQL Server)
-- SELECT TOP 5 * FROM Students ORDER BY CPI DESC;
```

The `SELECT` statement, with its powerful combination of clauses, allows you to retrieve highly specific subsets and summaries of data from your relational database.

---
**Related Notes:**
* [[SQL]]
* [[SQL DML]]
* [[SQL Joins]]
* [[SQL Aggregate Functions and Grouping]]
* [[SQL Subqueries]]
* [[SQL NULL Values]]