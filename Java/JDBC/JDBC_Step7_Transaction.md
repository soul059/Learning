# JDBC Step 7: Transaction Management

In database terms, a **transaction** is a sequence of one or more SQL operations that are executed as a single, atomic unit of work. This means that either **all** of the operations in the transaction succeed, or **none** of them do. If any single operation fails, the entire transaction is rolled back, and the database is left in the state it was in before the transaction began.

Transaction management is crucial for maintaining data integrity and consistency.

---

## The ACID Properties

Transactions are defined by four key properties, known as ACID:

1.  **Atomicity**: A transaction is an "all or nothing" operation. It cannot be partially completed.
2.  **Consistency**: A transaction must bring the database from one valid state to another. It must not violate any of the database's integrity constraints.
3.  **Isolation**: Transactions are often executed concurrently. Isolation ensures that the execution of one transaction is isolated from that of others. The intermediate state of a transaction is not visible to other transactions.
4.  **Durability**: Once a transaction has been successfully committed, the changes are permanent and will survive any subsequent system failure (like a power outage or crash).

---

## JDBC Transaction Control

By default, JDBC connections operate in **auto-commit mode**. In this mode, every single SQL statement is treated as its own transaction and is automatically committed to the database as soon as it's executed.

While simple, auto-commit mode is not suitable for operations that involve multiple related steps. Consider a bank transfer:
1.  Debit money from Account A.
2.  Credit money to Account B.

If the system crashes after step 1 but before step 2, the money has vanished from Account A but never appeared in Account B. This violates data integrity. To solve this, you must manage the transaction manually.

### Key `Connection` Methods for Transaction Management

The `java.sql.Connection` interface provides the methods needed to control transactions:

-   **`void setAutoCommit(boolean autoCommit)`**:
    -   `setAutoCommit(false)`: Disables auto-commit mode and marks the beginning of a transaction.
    -   `setAutoCommit(true)`: Re-enables auto-commit mode.

-   **`void commit()`**:
    -   Makes all changes made since the previous commit/rollback permanent in the database. This marks the end of the current transaction.

-   **`void rollback()`**:
    -   Discards all changes made in the current transaction and restores the database to the state it was in before the transaction started.

---

## Transaction Management Pattern

The standard pattern for manual transaction management involves a `try-catch` block.

1.  Disable auto-commit.
2.  Execute all your SQL statements inside the `try` block.
3.  If all statements execute without error, call `commit()` at the end of the `try` block.
4.  In the `catch` block, call `rollback()` to undo any partial changes if an exception occurs.
5.  In a `finally` block, it's good practice to restore the connection's original auto-commit state.

### Example: Bank Transfer

```java
Connection conn = null;
String debitSql = "UPDATE accounts SET balance = balance - ? WHERE id = ?";
String creditSql = "UPDATE accounts SET balance = balance + ? WHERE id = ?";

try {
    conn = DriverManager.getConnection(URL, USER, PASS);

    // 1. Disable auto-commit to start the transaction
    conn.setAutoCommit(false);

    // --- Start of Transaction ---

    // 2. Debit from Account 1
    try (PreparedStatement debitPstmt = conn.prepareStatement(debitSql)) {
        debitPstmt.setDouble(1, 100.00);
        debitPstmt.setInt(2, 1); // Account ID 1
        debitPstmt.executeUpdate();
    }

    // Simulate an error condition
    // if (true) { throw new SQLException("Simulated system failure!"); }

    // 3. Credit to Account 2
    try (PreparedStatement creditPstmt = conn.prepareStatement(creditSql)) {
        creditPstmt.setDouble(1, 100.00);
        creditPstmt.setInt(2, 2); // Account ID 2
        creditPstmt.executeUpdate();
    }

    // 4. If both operations succeed, commit the transaction
    conn.commit();
    System.out.println("Transaction committed successfully.");

    // --- End of Transaction ---

} catch (SQLException e) {
    System.err.println("Transaction failed. Rolling back...");
    e.printStackTrace();
    if (conn != null) {
        try {
            // 5. If any error occurs, roll back the entire transaction
            conn.rollback();
            System.err.println("Rollback successful.");
        } catch (SQLException ex) {
            System.err.println("Error during rollback.");
            ex.printStackTrace();
        }
    }
} finally {
    if (conn != null) {
        try {
            // 6. Always restore the original auto-commit state and close the connection
            conn.setAutoCommit(true);
            conn.close();
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
}
```
This pattern ensures that the bank transfer is **atomic**. Both the debit and credit operations will succeed together, or they will both fail together, leaving the database in a consistent state.
