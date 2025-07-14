# JDBC Step 5: The ResultSet

When you execute a `SELECT` query using `executeQuery()`, the database returns the results as a `java.sql.ResultSet`. This object is a table of data representing the rows and columns from your query result. It is the primary way you access the data retrieved from the database.

---

## What is a `ResultSet`?

A `ResultSet` object maintains a cursor that points to the current row of data. Initially, the cursor is positioned **before the first row**. You must move the cursor to a valid row before you can retrieve any data.

**Key Characteristics:**
-   **It's a cursor, not a data store**: The `ResultSet` doesn't necessarily hold all the data in memory. It can be a live view into the database, fetching rows from the server as needed. This behavior is driver-dependent.
-   **It needs to be closed**: A `ResultSet` holds database resources (like the cursor on the server side). It is crucial to close it when you are done to release these resources. The `try-with-resources` statement is the best way to manage this.

---

## Navigating the `ResultSet`

The most common way to process a `ResultSet` is to iterate through it row by row using a `while` loop and the `next()` method.

### `boolean next()`
-   This is the most important method. It moves the cursor one row forward from its current position.
-   It returns `true` if the new current row is valid (i.e., if there is another row to process).
-   It returns `false` if there are no more rows.

This makes it perfect for use in a `while` loop condition.

### Standard Processing Loop

```java
String sql = "SELECT id, name, price FROM products";
try (Statement stmt = connection.createStatement();
     ResultSet rs = stmt.executeQuery(sql)) { // try-with-resources for auto-closing

    // The loop continues as long as rs.next() is true
    while (rs.next()) {
        // Now the cursor is on a valid row. We can retrieve data.
        // ... processing logic for the current row ...
    }
    
} catch (SQLException e) {
    e.printStackTrace();
}
```

---

## Retrieving Data from a Row

Once the cursor is on a valid row, you can retrieve the value of each column using the various `getXxx()` methods, such as `getInt()`, `getString()`, `getDouble()`, `getDate()`, etc.

You can identify the column you want to retrieve in two ways:

1.  **By Column Name (Recommended)**: Use the `String` name of the column as it appears in your SQL query. This is more readable and less prone to breaking if the query changes.
    -   `rs.getString("name")`
    -   `rs.getDouble("price")`

2.  **By Column Index (1-based)**: Use the 1-based integer index of the column. The first column is index 1, the second is 2, and so on. This can be slightly more performant but is less readable and more brittle.
    -   `rs.getInt(1)` (for the `id` column)
    -   `rs.getString(2)` (for the `name` column)

### Example: Retrieving Data

```java
String sql = "SELECT id, name, launch_date, price FROM products WHERE category = 'Electronics'";

try (PreparedStatement pstmt = connection.prepareStatement(sql);
     ResultSet rs = pstmt.executeQuery()) {

    System.out.println("Electronic Products:");
    while (rs.next()) {
        // Retrieve data using column names
        int id = rs.getInt("id");
        String name = rs.getString("name");
        java.sql.Date launchDate = rs.getDate("launch_date");
        double price = rs.getDouble("price");

        // It's good practice to check for nulls for non-primitive types
        if (name == null) {
            name = "N/A";
        }

        System.out.printf("ID: %d, Name: %s, Date: %s, Price: %.2f%n",
                id, name, launchDate, price);
    }

} catch (SQLException e) {
    e.printStackTrace();
}
```

---

## Handling `NULL` Values

In SQL, a column can have a `NULL` value. When you retrieve this with a `getXxx()` method, the behavior depends on the return type:
-   For methods that return an object (e.g., `getString()`, `getDate()`), the result will be a Java `null`.
-   For methods that return a primitive (e.g., `getInt()`, `getDouble()`), the result will be `0` or `0.0`. This can be ambiguous.

To reliably check for a `NULL` value, you should use the `wasNull()` method. You must call `wasNull()` **immediately after** calling a `getXxx()` method.

### Example: Checking for `NULL`

```java
int managerId = rs.getInt("manager_id");
if (rs.wasNull()) {
    // The manager_id was NULL in the database
    System.out.println("This employee has no manager.");
} else {
    // The manager_id was a valid integer (which could be 0)
    System.out.println("Manager ID: " + managerId);
}
```

By effectively navigating and retrieving data from the `ResultSet`, you can bring the results of your database queries into your Java application for processing.
