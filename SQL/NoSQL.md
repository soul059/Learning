---
tags:
  - database
  - nosql
  - data-models
  - non-relational
---

# NoSQL (Not Only SQL)

**NoSQL** is a broad category of database management systems that do **not** primarily use the traditional tabular relations found in the [[Relational Model]]. While the term "NoSQL" originally meant "non SQL," it's more accurately understood as "Not Only SQL," as many NoSQL databases do support SQL-like query languages or integrate with SQL ecosystems.

NoSQL databases emerged in the early 21st century as a response to the challenges faced by traditional [[DBMS|RDBMS]] when dealing with:

* **Big Data:** Handling massive volumes of structured, semi-structured, and unstructured data.
* **Real-time Web Applications:** Requiring extremely high read/write throughput and low latency.
* **Rapid Development:** Needing flexible schemas that can evolve quickly.
* **Horizontal Scaling (Scale-out):** Easily distributing data and load across many servers or clusters, rather than relying on scaling up a single powerful server.

## Key Characteristics of NoSQL Databases

NoSQL databases often diverge from the [[Relational Model]] in several key ways:

* **Schema-less or Flexible Schema:** Unlike RDBMS which require a rigid, predefined schema, NoSQL databases often allow data to be added without first defining a strict table structure. This provides flexibility for evolving data requirements.
* **Horizontal Scaling:** Designed to scale out by adding more servers to a cluster, rather than scaling up a single server (Vertical Scaling). This is often handled automatically by the database system.
* **Diverse Data Models:** They utilize various data models beyond the relational table, tailored for specific types of data and access patterns.
* **Eventual Consistency:** Many NoSQL databases prioritize Availability and Partition Tolerance over immediate Consistency (following the [[CAP Theorem]] - create this note if needed), offering eventual consistency where data becomes consistent across the system over time, but may be temporarily inconsistent after updates. *Some NoSQL databases offer stronger consistency guarantees.*
* **Optimized for Specific Workloads:** Different types of NoSQL databases are optimized for different access patterns (e.g., key-value lookups, document retrieval, graph traversal).

## Types of NoSQL Databases

NoSQL databases are often categorized based on their underlying data model:

1.  **Key-Value Stores:**
    * **Model:** Data is stored as simple key-value pairs, similar to a hash map or dictionary. Values are typically opaque to the database.
    * **Use Cases:** Caching, session management, user profiles, leaderboards.
    * **Examples:** Redis, Memcached, Amazon DynamoDB, Riak.

2.  **Document Databases:**
    * **Model:** Data is stored in flexible, semi-structured documents (often in formats like JSON, BSON, or XML). Documents can have nested structures and varying fields.
    * **Use Cases:** Content management systems, catalogs, user profiles, mobile application backends.
    * **Examples:** MongoDB, Couchbase, Apache CouchDB, RavenDB.

3.  **Column-Family Stores (Wide-Column Stores):**
    * **Model:** Data is stored in tables, but rows can have different sets of columns within column families. Optimized for distributed storage and querying large amounts of data across columns.
    * **Use Cases:** Time-series data, logging, large-scale analytics, sensor data.
    * **Examples:** Apache Cassandra, HBase, Google Cloud Bigtable.

4.  **Graph Databases:**
    * **Model:** Data is stored as nodes (entities) and edges (relationships) between them. Optimized for managing and traversing highly interconnected data.
    * **Use Cases:** Social networks, recommendation engines, fraud detection, network topology.
    * **Examples:** Neo4j, ArangoDB, Amazon Neptune, JanusGraph.

## NoSQL vs. Relational Databases

| Feature           | Relational Databases (RDBMS)               | NoSQL Databases                                    |
| :---------------- | :----------------------------------------- | :------------------------------------------------- |
| **Data Model** | Tables (relations) with fixed schemas      | Diverse (Key-Value, Document, Column-Family, Graph), flexible/schema-less |
| **Schema** | Rigid, predefined                          | Flexible, dynamic, schema-on-read common           |
| **Scaling** | Primarily Vertical (scale up), Horizontal often complex | Primarily Horizontal (scale out)                   |
| **Consistency** | Strong Consistency (ACID)                  | Varying (often Eventual Consistency), ACID rare      |
| **Query Language**| [[SQL]]                                  | Varies widely (APIs, query languages specific to model) |
| **Relationships** | Defined via Foreign Keys (joins)           | Often embedded, linked, or graph traversals        |
| **Complexity** | Mature, well-understood, complex queries possible | Simpler data access patterns often, complex queries can be hard |

NoSQL databases are not intended to replace RDBMS entirely but are valuable alternatives for specific use cases where the flexibility and horizontal scalability they offer are more critical than the strict consistency and structured querying capabilities of the [[Relational Model]]. Many modern applications use a polyglot persistence approach, combining different types of databases (RDBMS and various NoSQL types) to meet different needs.

---
**Related Notes:**
* [[Database Concepts]]
* [[DBMS]]
* [[Data Models]]
* [[Relational Model]]
* [[CAP Theorem]] (Create if needed)
* [[Big Data]] (Create if needed)