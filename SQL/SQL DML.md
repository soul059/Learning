---
tags:
  - sql
  - dml
  - insert
  - update
  - delete
---

# SQL DML (Data Manipulation Language)

[[SQL|DML (Data Manipulation Language)]] commands are used to manage data within the database objects defined by [[SQL DDL|DDL]]. These commands are for querying, adding, deleting, and modifying the records (tuples) in your tables.

The primary DML commands are:

* `SELECT`: Retrieves data (often categorized separately as DQL).
* `INSERT`: Adds new rows of data.
* `UPDATE`: Modifies existing data.
* `DELETE`: Removes rows of data.

## SELECT (DQL - Data Query Language)

While strictly DQL, `SELECT` is fundamental to data manipulation as it's used to identify the data you want to work with (read, update, delete). Because of its complexity and importance, it's often discussed separately.

Refer to [[SQL DQL]] for a detailed explanation of the `SELECT` statement and its various clauses (`FROM`, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, etc.).

## INSERT

The `INSERT` statement is used to add new rows (tuples) to a table.

### Inserting a Single Row

```sql
INSERT INTO table_name (column1, column2, column3, ...)
VALUES (value1, value2, value3, ...);
```
* You list the columns you are providing values for. This is good practice as it makes the statement clearer and resilient to schema changes (like adding a new column with a default value).
* The number and order of values in the `VALUES` clause must match the columns listed.

If you are providing values for *all* columns in the table *in the order they were defined*, you can omit the column list:

```sql
INSERT INTO table_name
VALUES (value1, value2, value3, ...); -- Less recommended practice
```

**Example:**

```sql
INSERT INTO Students (StudentID, FirstName, LastName, MajorID)
VALUES (101, 'Alice', 'Smith', 1);
```

```sql
INSERT INTO Majors (MajorID, MajorName)
VALUES (1, 'Computer Science');
```

### Inserting Multiple Rows

Many SQL dialects allow inserting multiple rows in a single `INSERT` statement:

```sql
INSERT INTO table_name (column1, column2, ...)
VALUES
    (value1a, value2a, ...),
    (value1b, value2b, ...),
    (value1c, value2c, ...);
```

### Inserting from Another Table (INSERT...SELECT)

You can insert rows into a table by selecting data from another table using a `SELECT` statement.

```sql
INSERT INTO table_name (column1, column2, ...)
SELECT column1, column2, ...
FROM source_table
WHERE condition;
```

**Example:**

```sql
-- Assume a table 'NewStudents' with same relevant columns
INSERT INTO Students (StudentID, FirstName, LastName, MajorID)
SELECT StudentID, FirstName, LastName, MajorID
FROM NewStudents
WHERE EnrollmentDate %3E= '2024-09-01';
```
The columns selected from the `source_table` must match the columns specified in the `INSERT INTO` table in terms of count and compatible data types.

## UPDATE

The `UPDATE` statement is used to modify existing records in a table.

```sql
UPDATE table_name
SET column1 = new_value1,
    column2 = new_value2,
    ...
WHERE condition;
```

* `SET`: Specifies the columns to modify and their new values.
* `WHERE`: (Crucial!) Specifies which rows to update. **If you omit the `WHERE` clause, ALL rows in the table will be updated!**

**Example:**

```sql
-- Update the major for student with StudentID 101
UPDATE Students
SET MajorID = 2
WHERE StudentID = 101;
```

```sql
-- Give all Computer Science majors a default enrollment date (if it's null)
UPDATE Students
SET EnrollmentDate = '2024-09-01'
WHERE MajorID = (SELECT MajorID FROM Majors WHERE MajorName = 'Computer Science')
AND EnrollmentDate IS NULL;
```

## DELETE

The `DELETE` statement is used to remove one or more rows from a table.

```sql
DELETE FROM table_name
WHERE condition;
```

* `WHERE`: (Crucial!) Specifies which rows to delete. **If you omit the `WHERE` clause, ALL rows in the table will be deleted!** (Similar effect to `TRUNCATE TABLE`, but often slower and logs individual row deletions, making it [[SQL Transactions|transactionally]] safer).

**Example:**

```sql
-- Delete the student with StudentID 101
DELETE FROM Students
WHERE StudentID = 101;
```

```sql
-- Delete all students enrolled before 2020
DELETE FROM Students
WHERE EnrollmentDate %3C '2020-01-01';
```

DML commands are the tools you use daily to populate and maintain the data content of your relational database, working in conjunction with [[SQL DQL|SELECT]] to identify the target data.

---
**Related Notes:**
* [[SQL]]
* [[SQL DQL]]
* [[SQL DDL]]
* [[SQL Constraints]]
* [[SQL Transactions]]