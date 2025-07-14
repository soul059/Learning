---
tags:
  - sql
  - data-types
  - database-basics
---

# SQL Data Types

[[SQL|SQL]] data types define the type of data that can be stored in a column. Choosing the correct data type is important for data integrity, storage efficiency, and performance. While specific implementations vary slightly between RDBMS, here are common categories and examples:

## Numeric Data Types

Used for storing numeric values.

* **Integer Types:** Whole numbers.
    * `INT` or `INTEGER`: Standard integer.
    * `SMALLINT`: Smaller range than INT.
    * `BIGINT`: Larger range than INT.
    * `TINYINT`: Very small range (often 0-255 or -128 to 127).
    * *Variations exist (e.g., `MEDIUMINT` in MySQL).*
* **Decimal Types:** Exact numeric values with a fixed precision and scale.
    * `DECIMAL(p, s)` or `NUMERIC(p, s)`: `p` is total number of digits (precision), `s` is number of digits after decimal point (scale). `DECIMAL` is preferred for financial data where exactness is crucial.
* **Floating-Point Types:** Approximate numeric values.
    * `FLOAT(n)`: Floating-point number with at least `n` bits of precision.
    * `REAL`: Typically a single-precision floating-point.
    * `DOUBLE PRECISION` or `DOUBLE`: Typically a double-precision floating-point.
    * *Use with caution for financial or exact calculations due to potential precision issues.*

## String (Character) Data Types

Used for storing text and character strings.

* **Fixed-Length:** Stores a fixed number of characters. Padded with spaces if the string is shorter than the defined length.
    * `CHAR(n)`: Fixed-length string of `n` characters.
* **Variable-Length:** Stores strings up to a maximum defined length. Only uses space required for the actual string (plus a small overhead).
    * `VARCHAR(n)` or `CHARACTER VARYING(n)`: Variable-length string up to `n` characters.
    * `TEXT`: Can store very long strings (often without a specified max length, or a very large default).
* **Binary String Types:** For storing binary data (like images, audio).
    * `BINARY(n)`: Fixed-length binary string.
    * `VARBINARY(n)`: Variable-length binary string.
    * `BLOB` (Binary Large Object): For very large binary data.

## Date and Time Data Types

Used for storing date and time values.

* `DATE`: Stores a date (year, month, day).
* `TIME`: Stores a time of day (hour, minute, second, optionally fractions of a second).
* `DATETIME` or `TIMESTAMP`: Stores both date and time. `TIMESTAMP` often includes timezone information or represents a point in time independent of timezone.
* `YEAR`: Stores a year.

## Boolean Data Types

Used for storing truth values.

* `BOOLEAN` or `BOOL`: Stores `TRUE`, `FALSE`, or `NULL`. *Support varies; some RDBMS use `TINYINT` (0 or 1) instead.*

## Other Data Types

* `NULL`: Not a data type itself, but a special marker indicating missing or unknown data. (See [[SQL NULL Values]])
* `ENUM`: (Common in MySQL) Allows a column to have only one value from a predefined list of permitted values.
* `SET`: (Common in MySQL) Allows a column to have zero or more values from a predefined list of permitted values.
* `JSON`: (Increasingly supported) For storing JSON documents.
* Spatial Data Types: For geographical or geometric data.

Choosing the right data type is part of [[Database Design]] and implemented using [[SQL DDL]] statements like `CREATE TABLE`.

---
_**Related Notes:**_
- [[SQL]]
- [[SQL DDL]]
- [[SQL Constraints]]
- [[SQL NULL Values]]