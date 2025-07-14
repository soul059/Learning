# Networking in Java

## 1. Socket Programming Basics

### TCP Sockets (Reliable Communication)

#### Simple TCP Server
```java
import java.io.*;
import java.net.*;
import java.util.concurrent.*;

public class SimpleTCPServer {
    private static final int PORT = 8080;
    private static final int MAX_CLIENTS = 10;
    
    public void startServer() {
        ExecutorService executor = Executors.newFixedThreadPool(MAX_CLIENTS);
        
        try (ServerSocket serverSocket = new ServerSocket(PORT)) {
            System.out.println("Server started on port " + PORT);
            System.out.println("Waiting for client connections...");
            
            while (true) {
                Socket clientSocket = serverSocket.accept();
                System.out.println("New client connected: " + clientSocket.getInetAddress());
                
                // Handle each client in a separate thread
                executor.submit(new ClientHandler(clientSocket));
            }
            
        } catch (IOException e) {
            System.out.println("Server error: " + e.getMessage());
        } finally {
            executor.shutdown();
        }
    }
    
    // Inner class to handle individual client connections
    private static class ClientHandler implements Runnable {
        private Socket clientSocket;
        private PrintWriter out;
        private BufferedReader in;
        
        public ClientHandler(Socket socket) {
            this.clientSocket = socket;
        }
        
        @Override
        public void run() {
            try {
                // Set up I/O streams
                out = new PrintWriter(clientSocket.getOutputStream(), true);
                in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
                
                // Send welcome message
                out.println("Welcome to the server!");
                out.println("Type 'bye' to disconnect.");
                
                String inputLine;
                while ((inputLine = in.readLine()) != null) {
                    System.out.println("Client says: " + inputLine);
                    
                    if ("bye".equalsIgnoreCase(inputLine.trim())) {
                        out.println("Goodbye!");
                        break;
                    }
                    
                    // Echo the message back
                    out.println("Echo: " + inputLine);
                }
                
            } catch (IOException e) {
                System.out.println("Error handling client: " + e.getMessage());
            } finally {
                try {
                    if (out != null) out.close();
                    if (in != null) in.close();
                    clientSocket.close();
                    System.out.println("Client disconnected: " + clientSocket.getInetAddress());
                } catch (IOException e) {
                    System.out.println("Error closing client connection: " + e.getMessage());
                }
            }
        }
    }
    
    public static void main(String[] args) {
        new SimpleTCPServer().startServer();
    }
}
```

#### Simple TCP Client
```java
import java.io.*;
import java.net.*;
import java.util.Scanner;

public class SimpleTCPClient {
    private static final String SERVER_HOST = "localhost";
    private static final int SERVER_PORT = 8080;
    
    public void connectToServer() {
        try (Socket socket = new Socket(SERVER_HOST, SERVER_PORT);
             PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
             BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
             Scanner scanner = new Scanner(System.in)) {
            
            System.out.println("Connected to server: " + SERVER_HOST + ":" + SERVER_PORT);
            
            // Start a thread to read server responses
            Thread readerThread = new Thread(() -> {
                try {
                    String serverResponse;
                    while ((serverResponse = in.readLine()) != null) {
                        System.out.println("Server: " + serverResponse);
                    }
                } catch (IOException e) {
                    System.out.println("Connection to server lost");
                }
            });
            readerThread.setDaemon(true);
            readerThread.start();
            
            // Main thread handles user input
            System.out.println("Enter messages (type 'bye' to quit):");
            String userInput;
            while (!(userInput = scanner.nextLine()).equalsIgnoreCase("bye")) {
                out.println(userInput);
            }
            
            out.println("bye"); // Send bye to server
            
        } catch (IOException e) {
            System.out.println("Client error: " + e.getMessage());
        }
    }
    
    public static void main(String[] args) {
        new SimpleTCPClient().connectToServer();
    }
}
```

### UDP Sockets (Fast, Unreliable Communication)

#### UDP Server
```java
import java.net.*;
import java.io.IOException;

public class SimpleUDPServer {
    private static final int PORT = 9090;
    private static final int BUFFER_SIZE = 1024;
    
    public void startServer() {
        try (DatagramSocket socket = new DatagramSocket(PORT)) {
            System.out.println("UDP Server started on port " + PORT);
            
            byte[] buffer = new byte[BUFFER_SIZE];
            
            while (true) {
                // Create packet to receive data
                DatagramPacket packet = new DatagramPacket(buffer, buffer.length);
                
                // Receive data
                socket.receive(packet);
                
                // Extract message
                String message = new String(packet.getData(), 0, packet.getLength());
                System.out.println("Received from " + packet.getAddress() + ":" + packet.getPort() + " - " + message);
                
                // Prepare response
                String response = "Echo: " + message;
                byte[] responseData = response.getBytes();
                
                // Send response back to client
                DatagramPacket responsePacket = new DatagramPacket(
                    responseData, 
                    responseData.length, 
                    packet.getAddress(), 
                    packet.getPort()
                );
                
                socket.send(responsePacket);
            }
            
        } catch (IOException e) {
            System.out.println("UDP Server error: " + e.getMessage());
        }
    }
    
    public static void main(String[] args) {
        new SimpleUDPServer().startServer();
    }
}
```

#### UDP Client
```java
import java.net.*;
import java.io.IOException;
import java.util.Scanner;

public class SimpleUDPClient {
    private static final String SERVER_HOST = "localhost";
    private static final int SERVER_PORT = 9090;
    private static final int BUFFER_SIZE = 1024;
    
    public void sendMessages() {
        try (DatagramSocket socket = new DatagramSocket();
             Scanner scanner = new Scanner(System.in)) {
            
            InetAddress serverAddress = InetAddress.getByName(SERVER_HOST);
            System.out.println("UDP Client ready. Enter messages (type 'quit' to exit):");
            
            String message;
            while (!(message = scanner.nextLine()).equalsIgnoreCase("quit")) {
                // Send message to server
                byte[] messageData = message.getBytes();
                DatagramPacket packet = new DatagramPacket(
                    messageData, 
                    messageData.length, 
                    serverAddress, 
                    SERVER_PORT
                );
                
                socket.send(packet);
                
                // Receive response
                byte[] buffer = new byte[BUFFER_SIZE];
                DatagramPacket responsePacket = new DatagramPacket(buffer, buffer.length);
                socket.receive(responsePacket);
                
                String response = new String(responsePacket.getData(), 0, responsePacket.getLength());
                System.out.println("Server response: " + response);
            }
            
        } catch (IOException e) {
            System.out.println("UDP Client error: " + e.getMessage());
        }
    }
    
    public static void main(String[] args) {
        new SimpleUDPClient().sendMessages();
    }
}
```

## 2. Advanced Socket Programming

### Multi-threaded Chat Server
```java
import java.io.*;
import java.net.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class ChatServer {
    private static final int PORT = 8888;
    private static Map<String, ClientHandler> clients = new ConcurrentHashMap<>();
    
    public void startServer() {
        try (ServerSocket serverSocket = new ServerSocket(PORT)) {
            System.out.println("Chat Server started on port " + PORT);
            
            while (true) {
                Socket clientSocket = serverSocket.accept();
                ClientHandler clientHandler = new ClientHandler(clientSocket);
                new Thread(clientHandler).start();
            }
            
        } catch (IOException e) {
            System.out.println("Chat Server error: " + e.getMessage());
        }
    }
    
    // Broadcast message to all connected clients
    public static void broadcastMessage(String message, String senderName) {
        for (Map.Entry<String, ClientHandler> entry : clients.entrySet()) {
            if (!entry.getKey().equals(senderName)) {
                entry.getValue().sendMessage(message);
            }
        }
    }
    
    // Send private message to specific client
    public static void sendPrivateMessage(String message, String targetUser, String senderName) {
        ClientHandler targetClient = clients.get(targetUser);
        if (targetClient != null) {
            targetClient.sendMessage("[Private from " + senderName + "]: " + message);
        } else {
            ClientHandler sender = clients.get(senderName);
            if (sender != null) {
                sender.sendMessage("User '" + targetUser + "' not found.");
            }
        }
    }
    
    // Get list of connected users
    public static String getConnectedUsers() {
        return "Connected users: " + String.join(", ", clients.keySet());
    }
    
    private static class ClientHandler implements Runnable {
        private Socket clientSocket;
        private PrintWriter out;
        private BufferedReader in;
        private String clientName;
        
        public ClientHandler(Socket socket) {
            this.clientSocket = socket;
        }
        
        @Override
        public void run() {
            try {
                out = new PrintWriter(clientSocket.getOutputStream(), true);
                in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
                
                // Get client name
                out.println("Enter your name:");
                clientName = in.readLine();
                
                if (clientName == null || clientName.trim().isEmpty()) {
                    out.println("Invalid name. Disconnecting.");
                    return;
                }
                
                clientName = clientName.trim();
                
                // Check if name is already taken
                if (clients.containsKey(clientName)) {
                    out.println("Name already taken. Disconnecting.");
                    return;
                }
                
                // Add client to the map
                clients.put(clientName, this);
                
                out.println("Welcome to the chat, " + clientName + "!");
                out.println("Commands: /users (list users), /private <user> <message> (private message), /quit (exit)");
                
                // Notify others about new user
                broadcastMessage(clientName + " joined the chat.", clientName);
                
                String inputLine;
                while ((inputLine = in.readLine()) != null) {
                    if (inputLine.startsWith("/quit")) {
                        break;
                    } else if (inputLine.startsWith("/users")) {
                        out.println(getConnectedUsers());
                    } else if (inputLine.startsWith("/private ")) {
                        handlePrivateMessage(inputLine);
                    } else {
                        // Broadcast public message
                        broadcastMessage("[" + clientName + "]: " + inputLine, clientName);
                    }
                }
                
            } catch (IOException e) {
                System.out.println("Error handling client " + clientName + ": " + e.getMessage());
            } finally {
                cleanup();
            }
        }
        
        private void handlePrivateMessage(String input) {
            String[] parts = input.split(" ", 3);
            if (parts.length >= 3) {
                String targetUser = parts[1];
                String message = parts[2];
                sendPrivateMessage(message, targetUser, clientName);
            } else {
                out.println("Usage: /private <username> <message>");
            }
        }
        
        public void sendMessage(String message) {
            if (out != null) {
                out.println(message);
            }
        }
        
        private void cleanup() {
            try {
                if (clientName != null) {
                    clients.remove(clientName);
                    broadcastMessage(clientName + " left the chat.", clientName);
                    System.out.println(clientName + " disconnected");
                }
                
                if (out != null) out.close();
                if (in != null) in.close();
                if (clientSocket != null) clientSocket.close();
                
            } catch (IOException e) {
                System.out.println("Error during cleanup: " + e.getMessage());
            }
        }
    }
    
    public static void main(String[] args) {
        new ChatServer().startServer();
    }
}
```

### Chat Client
```java
import java.io.*;
import java.net.*;
import java.util.Scanner;

public class ChatClient {
    private static final String SERVER_HOST = "localhost";
    private static final int SERVER_PORT = 8888;
    
    public void connectToChat() {
        try (Socket socket = new Socket(SERVER_HOST, SERVER_PORT);
             PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
             BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
             Scanner scanner = new Scanner(System.in)) {
            
            System.out.println("Connected to chat server");
            
            // Start thread to read server messages
            Thread readerThread = new Thread(() -> {
                try {
                    String serverMessage;
                    while ((serverMessage = in.readLine()) != null) {
                        System.out.println(serverMessage);
                    }
                } catch (IOException e) {
                    System.out.println("Connection to server lost");
                }
            });
            readerThread.setDaemon(true);
            readerThread.start();
            
            // Main thread handles user input
            String userInput;
            while ((userInput = scanner.nextLine()) != null) {
                out.println(userInput);
                
                if (userInput.startsWith("/quit")) {
                    break;
                }
            }
            
        } catch (IOException e) {
            System.out.println("Chat client error: " + e.getMessage());
        }
    }
    
    public static void main(String[] args) {
        new ChatClient().connectToChat();
    }
}
```

## 3. HTTP Client Programming

### Basic HTTP Client (Java 11+)
```java
import java.net.http.*;
import java.net.URI;
import java.time.Duration;
import java.util.concurrent.CompletableFuture;

public class ModernHTTPClient {
    
    public void synchronousRequests() {
        HttpClient client = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(10))
            .build();
        
        try {
            // GET request
            HttpRequest getRequest = HttpRequest.newBuilder()
                .uri(URI.create("https://jsonplaceholder.typicode.com/posts/1"))
                .GET()
                .build();
            
            HttpResponse<String> getResponse = client.send(getRequest, 
                HttpResponse.BodyHandlers.ofString());
            
            System.out.println("GET Response:");
            System.out.println("Status Code: " + getResponse.statusCode());
            System.out.println("Body: " + getResponse.body());
            
            // POST request
            String jsonData = "{\"title\":\"Test Post\",\"body\":\"This is a test\",\"userId\":1}";
            
            HttpRequest postRequest = HttpRequest.newBuilder()
                .uri(URI.create("https://jsonplaceholder.typicode.com/posts"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonData))
                .build();
            
            HttpResponse<String> postResponse = client.send(postRequest, 
                HttpResponse.BodyHandlers.ofString());
            
            System.out.println("\nPOST Response:");
            System.out.println("Status Code: " + postResponse.statusCode());
            System.out.println("Body: " + postResponse.body());
            
        } catch (Exception e) {
            System.out.println("HTTP request error: " + e.getMessage());
        }
    }
    
    public void asynchronousRequests() {
        HttpClient client = HttpClient.newHttpClient();
        
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create("https://jsonplaceholder.typicode.com/posts"))
            .GET()
            .build();
        
        // Asynchronous request
        CompletableFuture<HttpResponse<String>> futureResponse = 
            client.sendAsync(request, HttpResponse.BodyHandlers.ofString());
        
        System.out.println("Request sent asynchronously...");
        
        futureResponse
            .thenApply(HttpResponse::body)
            .thenAccept(body -> {
                System.out.println("Async response received:");
                System.out.println("Body length: " + body.length() + " characters");
                System.out.println("First 100 characters: " + 
                    body.substring(0, Math.min(100, body.length())));
            })
            .exceptionally(throwable -> {
                System.out.println("Error in async request: " + throwable.getMessage());
                return null;
            });
        
        // Wait for completion
        try {
            Thread.sleep(5000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
    
    public void customConfiguration() {
        HttpClient client = HttpClient.newBuilder()
            .version(HttpClient.Version.HTTP_2)
            .connectTimeout(Duration.ofSeconds(10))
            .followRedirects(HttpClient.Redirect.NORMAL)
            .build();
        
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create("https://httpbin.org/user-agent"))
            .header("User-Agent", "Java HTTP Client")
            .header("Accept", "application/json")
            .timeout(Duration.ofSeconds(30))
            .GET()
            .build();
        
        try {
            HttpResponse<String> response = client.send(request, 
                HttpResponse.BodyHandlers.ofString());
            
            System.out.println("Custom configuration response:");
            System.out.println("Status: " + response.statusCode());
            System.out.println("Headers: " + response.headers().map());
            System.out.println("Body: " + response.body());
            
        } catch (Exception e) {
            System.out.println("Custom request error: " + e.getMessage());
        }
    }
    
    public static void main(String[] args) {
        ModernHTTPClient httpClient = new ModernHTTPClient();
        
        httpClient.synchronousRequests();
        System.out.println("\n" + "=".repeat(50) + "\n");
        
        httpClient.asynchronousRequests();
        System.out.println("\n" + "=".repeat(50) + "\n");
        
        httpClient.customConfiguration();
    }
}
```

### Legacy HTTP Client (URLConnection)
```java
import java.io.*;
import java.net.*;
import java.util.Map;

public class LegacyHTTPClient {
    
    public void getRequest() {
        try {
            URL url = new URL("https://jsonplaceholder.typicode.com/posts/1");
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            
            // Configure request
            connection.setRequestMethod("GET");
            connection.setRequestProperty("Accept", "application/json");
            connection.setConnectTimeout(10000); // 10 seconds
            connection.setReadTimeout(30000);    // 30 seconds
            
            // Get response
            int responseCode = connection.getResponseCode();
            System.out.println("Response Code: " + responseCode);
            
            // Read response body
            BufferedReader reader;
            if (responseCode >= 200 && responseCode < 300) {
                reader = new BufferedReader(new InputStreamReader(connection.getInputStream()));
            } else {
                reader = new BufferedReader(new InputStreamReader(connection.getErrorStream()));
            }
            
            String line;
            StringBuilder response = new StringBuilder();
            while ((line = reader.readLine()) != null) {
                response.append(line).append("\n");
            }
            reader.close();
            
            System.out.println("Response Body:");
            System.out.println(response.toString());
            
            // Get headers
            System.out.println("\nResponse Headers:");
            Map<String, java.util.List<String>> headers = connection.getHeaderFields();
            for (Map.Entry<String, java.util.List<String>> entry : headers.entrySet()) {
                System.out.println(entry.getKey() + ": " + entry.getValue());
            }
            
            connection.disconnect();
            
        } catch (IOException e) {
            System.out.println("GET request error: " + e.getMessage());
        }
    }
    
    public void postRequest() {
        try {
            URL url = new URL("https://jsonplaceholder.typicode.com/posts");
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            
            // Configure request
            connection.setRequestMethod("POST");
            connection.setRequestProperty("Content-Type", "application/json");
            connection.setRequestProperty("Accept", "application/json");
            connection.setDoOutput(true);
            
            // Prepare data
            String jsonData = "{\"title\":\"Legacy HTTP Post\",\"body\":\"Test content\",\"userId\":1}";
            
            // Send data
            try (OutputStream os = connection.getOutputStream();
                 BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(os, "UTF-8"))) {
                writer.write(jsonData);
                writer.flush();
            }
            
            // Get response
            int responseCode = connection.getResponseCode();
            System.out.println("POST Response Code: " + responseCode);
            
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(connection.getInputStream()))) {
                
                String line;
                StringBuilder response = new StringBuilder();
                while ((line = reader.readLine()) != null) {
                    response.append(line).append("\n");
                }
                
                System.out.println("POST Response Body:");
                System.out.println(response.toString());
            }
            
            connection.disconnect();
            
        } catch (IOException e) {
            System.out.println("POST request error: " + e.getMessage());
        }
    }
    
    public static void main(String[] args) {
        LegacyHTTPClient client = new LegacyHTTPClient();
        
        client.getRequest();
        System.out.println("\n" + "=".repeat(50) + "\n");
        client.postRequest();
    }
}
```

## 4. NIO Socket Programming

### NIO Server
```java
import java.nio.*;
import java.nio.channels.*;
import java.net.*;
import java.io.IOException;
import java.util.*;

public class NIOServer {
    private static final int PORT = 8080;
    private Selector selector;
    private ServerSocketChannel serverChannel;
    
    public void startServer() {
        try {
            // Create selector and server channel
            selector = Selector.open();
            serverChannel = ServerSocketChannel.open();
            
            // Configure server channel
            serverChannel.configureBlocking(false);
            serverChannel.bind(new InetSocketAddress(PORT));
            serverChannel.register(selector, SelectionKey.OP_ACCEPT);
            
            System.out.println("NIO Server started on port " + PORT);
            
            while (true) {
                // Wait for events
                int readyChannels = selector.select();
                
                if (readyChannels == 0) {
                    continue;
                }
                
                Set<SelectionKey> selectedKeys = selector.selectedKeys();
                Iterator<SelectionKey> keyIterator = selectedKeys.iterator();
                
                while (keyIterator.hasNext()) {
                    SelectionKey key = keyIterator.next();
                    
                    if (key.isAcceptable()) {
                        handleAccept(key);
                    } else if (key.isReadable()) {
                        handleRead(key);
                    }
                    
                    keyIterator.remove();
                }
            }
            
        } catch (IOException e) {
            System.out.println("NIO Server error: " + e.getMessage());
        } finally {
            closeServer();
        }
    }
    
    private void handleAccept(SelectionKey key) throws IOException {
        ServerSocketChannel serverChannel = (ServerSocketChannel) key.channel();
        SocketChannel clientChannel = serverChannel.accept();
        
        if (clientChannel != null) {
            clientChannel.configureBlocking(false);
            clientChannel.register(selector, SelectionKey.OP_READ);
            
            System.out.println("New client connected: " + clientChannel.getRemoteAddress());
            
            // Send welcome message
            String welcomeMessage = "Welcome to NIO Server!\n";
            ByteBuffer buffer = ByteBuffer.wrap(welcomeMessage.getBytes());
            clientChannel.write(buffer);
        }
    }
    
    private void handleRead(SelectionKey key) throws IOException {
        SocketChannel clientChannel = (SocketChannel) key.channel();
        ByteBuffer buffer = ByteBuffer.allocate(1024);
        
        try {
            int bytesRead = clientChannel.read(buffer);
            
            if (bytesRead > 0) {
                buffer.flip();
                
                // Extract message
                byte[] data = new byte[buffer.remaining()];
                buffer.get(data);
                String message = new String(data).trim();
                
                System.out.println("Received from " + clientChannel.getRemoteAddress() + ": " + message);
                
                // Echo message back
                String response = "Echo: " + message + "\n";
                ByteBuffer responseBuffer = ByteBuffer.wrap(response.getBytes());
                clientChannel.write(responseBuffer);
                
                if ("bye".equalsIgnoreCase(message)) {
                    clientChannel.close();
                    key.cancel();
                    System.out.println("Client disconnected: " + clientChannel.getRemoteAddress());
                }
                
            } else if (bytesRead == -1) {
                // Client disconnected
                System.out.println("Client disconnected: " + clientChannel.getRemoteAddress());
                clientChannel.close();
                key.cancel();
            }
            
        } catch (IOException e) {
            System.out.println("Error reading from client: " + e.getMessage());
            clientChannel.close();
            key.cancel();
        }
    }
    
    private void closeServer() {
        try {
            if (selector != null) selector.close();
            if (serverChannel != null) serverChannel.close();
        } catch (IOException e) {
            System.out.println("Error closing server: " + e.getMessage());
        }
    }
    
    public static void main(String[] args) {
        new NIOServer().startServer();
    }
}
```

### NIO Client
```java
import java.nio.*;
import java.nio.channels.*;
import java.net.*;
import java.io.IOException;
import java.util.Scanner;

public class NIOClient {
    private static final String SERVER_HOST = "localhost";
    private static final int SERVER_PORT = 8080;
    
    public void connectToServer() {
        try (SocketChannel channel = SocketChannel.open();
             Scanner scanner = new Scanner(System.in)) {
            
            // Connect to server
            channel.connect(new InetSocketAddress(SERVER_HOST, SERVER_PORT));
            System.out.println("Connected to NIO server");
            
            // Configure non-blocking mode for reading
            channel.configureBlocking(false);
            
            // Start thread to read server responses
            Thread readerThread = new Thread(() -> {
                ByteBuffer buffer = ByteBuffer.allocate(1024);
                
                while (channel.isConnected()) {
                    try {
                        buffer.clear();
                        int bytesRead = channel.read(buffer);
                        
                        if (bytesRead > 0) {
                            buffer.flip();
                            byte[] data = new byte[buffer.remaining()];
                            buffer.get(data);
                            System.out.print("Server: " + new String(data));
                        }
                        
                        Thread.sleep(100); // Small delay to prevent busy waiting
                        
                    } catch (IOException | InterruptedException e) {
                        break;
                    }
                }
            });
            
            readerThread.setDaemon(true);
            readerThread.start();
            
            // Main thread handles user input
            System.out.println("Enter messages (type 'bye' to quit):");
            
            String userInput;
            while (!(userInput = scanner.nextLine()).equalsIgnoreCase("bye")) {
                // Send message to server
                ByteBuffer buffer = ByteBuffer.wrap((userInput + "\n").getBytes());
                
                // Ensure all data is written
                while (buffer.hasRemaining()) {
                    channel.write(buffer);
                }
            }
            
            // Send bye message
            ByteBuffer byeBuffer = ByteBuffer.wrap("bye\n".getBytes());
            while (byeBuffer.hasRemaining()) {
                channel.write(byeBuffer);
            }
            
            Thread.sleep(1000); // Give time for server response
            
        } catch (IOException | InterruptedException e) {
            System.out.println("NIO Client error: " + e.getMessage());
        }
    }
    
    public static void main(String[] args) {
        new NIOClient().connectToServer();
    }
}
```

## 5. Network Utilities and Information

### Network Information
```java
import java.net.*;
import java.util.*;

public class NetworkUtilities {
    
    public void getNetworkInterfaces() {
        try {
            System.out.println("Network Interfaces:");
            System.out.println("=".repeat(50));
            
            Enumeration<NetworkInterface> interfaces = NetworkInterface.getNetworkInterfaces();
            
            while (interfaces.hasMoreElements()) {
                NetworkInterface netInterface = interfaces.nextElement();
                
                System.out.println("Interface: " + netInterface.getName());
                System.out.println("Display Name: " + netInterface.getDisplayName());
                System.out.println("Is Up: " + netInterface.isUp());
                System.out.println("Is Loopback: " + netInterface.isLoopback());
                System.out.println("Is Virtual: " + netInterface.isVirtual());
                System.out.println("Supports Multicast: " + netInterface.supportsMulticast());
                
                // Get MAC address
                byte[] mac = netInterface.getHardwareAddress();
                if (mac != null) {
                    StringBuilder macAddress = new StringBuilder();
                    for (int i = 0; i < mac.length; i++) {
                        macAddress.append(String.format("%02X%s", mac[i], (i < mac.length - 1) ? ":" : ""));
                    }
                    System.out.println("MAC Address: " + macAddress.toString());
                }
                
                // Get IP addresses
                Enumeration<InetAddress> addresses = netInterface.getInetAddresses();
                while (addresses.hasMoreElements()) {
                    InetAddress address = addresses.nextElement();
                    System.out.println("IP Address: " + address.getHostAddress() + 
                                     " (Type: " + (address instanceof Inet4Address ? "IPv4" : "IPv6") + ")");
                }
                
                System.out.println("-".repeat(30));
            }
            
        } catch (SocketException e) {
            System.out.println("Error getting network interfaces: " + e.getMessage());
        }
    }
    
    public void getHostInformation() {
        try {
            System.out.println("\nHost Information:");
            System.out.println("=".repeat(50));
            
            // Local host information
            InetAddress localhost = InetAddress.getLocalHost();
            System.out.println("Local Host Name: " + localhost.getHostName());
            System.out.println("Local Host Address: " + localhost.getHostAddress());
            
            // Get all local addresses
            InetAddress[] allLocalAddresses = InetAddress.getAllByName(localhost.getHostName());
            System.out.println("All Local Addresses:");
            for (InetAddress addr : allLocalAddresses) {
                System.out.println("  " + addr.getHostAddress());
            }
            
            // DNS lookup examples
            System.out.println("\nDNS Lookups:");
            String[] hostnames = {"google.com", "github.com", "stackoverflow.com"};
            
            for (String hostname : hostnames) {
                try {
                    InetAddress[] addresses = InetAddress.getAllByName(hostname);
                    System.out.println(hostname + ":");
                    for (InetAddress addr : addresses) {
                        System.out.println("  " + addr.getHostAddress());
                    }
                } catch (UnknownHostException e) {
                    System.out.println(hostname + ": Unable to resolve");
                }
            }
            
        } catch (UnknownHostException e) {
            System.out.println("Error getting host information: " + e.getMessage());
        }
    }
    
    public void portScanner(String host, int startPort, int endPort) {
        System.out.println("\nPort Scanner for " + host + " (ports " + startPort + "-" + endPort + "):");
        System.out.println("=".repeat(50));
        
        List<Integer> openPorts = new ArrayList<>();
        
        for (int port = startPort; port <= endPort; port++) {
            try (Socket socket = new Socket()) {
                socket.connect(new InetSocketAddress(host, port), 1000); // 1 second timeout
                openPorts.add(port);
                System.out.println("Port " + port + ": OPEN");
            } catch (IOException e) {
                // Port is closed or filtered
            }
        }
        
        if (openPorts.isEmpty()) {
            System.out.println("No open ports found in the specified range.");
        } else {
            System.out.println("\nSummary - Open ports: " + openPorts);
        }
    }
    
    public void urlOperations() {
        System.out.println("\nURL Operations:");
        System.out.println("=".repeat(50));
        
        try {
            URL url = new URL("https://www.example.com:8080/path/to/resource?param1=value1&param2=value2#section");
            
            System.out.println("Full URL: " + url.toString());
            System.out.println("Protocol: " + url.getProtocol());
            System.out.println("Host: " + url.getHost());
            System.out.println("Port: " + url.getPort());
            System.out.println("Default Port: " + url.getDefaultPort());
            System.out.println("Path: " + url.getPath());
            System.out.println("Query: " + url.getQuery());
            System.out.println("Fragment: " + url.getRef());
            System.out.println("User Info: " + url.getUserInfo());
            System.out.println("Authority: " + url.getAuthority());
            
            // URI operations
            URI uri = url.toURI();
            System.out.println("\nURI Operations:");
            System.out.println("Scheme: " + uri.getScheme());
            System.out.println("Authority: " + uri.getAuthority());
            System.out.println("Path: " + uri.getPath());
            System.out.println("Is Absolute: " + uri.isAbsolute());
            System.out.println("Is Opaque: " + uri.isOpaque());
            
        } catch (Exception e) {
            System.out.println("Error with URL operations: " + e.getMessage());
        }
    }
    
    public static void main(String[] args) {
        NetworkUtilities netUtils = new NetworkUtilities();
        
        netUtils.getNetworkInterfaces();
        netUtils.getHostInformation();
        netUtils.urlOperations();
        
        // Uncomment to run port scanner (be careful with target host)
        // netUtils.portScanner("localhost", 8000, 8100);
    }
}
```

## 6. SSL/TLS Secure Networking

### SSL Client
```java
import javax.net.ssl.*;
import java.io.*;
import java.security.cert.X509Certificate;

public class SSLClient {
    
    public void simpleSSLConnection() {
        try {
            // Create SSL context
            SSLContext sslContext = SSLContext.getInstance("TLS");
            sslContext.init(null, null, null); // Use default trust store
            
            SSLSocketFactory factory = sslContext.getSocketFactory();
            
            try (SSLSocket socket = (SSLSocket) factory.createSocket("www.google.com", 443);
                 PrintWriter out = new PrintWriter(socket.getOutputStream());
                 BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()))) {
                
                // Send HTTP request
                out.println("GET / HTTP/1.1");
                out.println("Host: www.google.com");
                out.println("Connection: close");
                out.println();
                out.flush();
                
                // Read response
                String line;
                int lineCount = 0;
                while ((line = in.readLine()) != null && lineCount < 20) {
                    System.out.println(line);
                    lineCount++;
                }
                
                // Get SSL session information
                SSLSession session = socket.getSession();
                System.out.println("\nSSL Session Information:");
                System.out.println("Protocol: " + session.getProtocol());
                System.out.println("Cipher Suite: " + session.getCipherSuite());
                System.out.println("Peer Host: " + session.getPeerHost());
                System.out.println("Peer Port: " + session.getPeerPort());
                
            }
            
        } catch (Exception e) {
            System.out.println("SSL connection error: " + e.getMessage());
        }
    }
    
    public void trustAllCertificates() {
        try {
            // Create a trust manager that accepts all certificates (NOT for production!)
            TrustManager[] trustAllCerts = new TrustManager[] {
                new X509TrustManager() {
                    public X509Certificate[] getAcceptedIssuers() {
                        return null;
                    }
                    
                    public void checkClientTrusted(X509Certificate[] certs, String authType) {
                        // Do nothing - trust all
                    }
                    
                    public void checkServerTrusted(X509Certificate[] certs, String authType) {
                        // Do nothing - trust all
                    }
                }
            };
            
            SSLContext sslContext = SSLContext.getInstance("TLS");
            sslContext.init(null, trustAllCerts, null);
            
            // Set default SSL socket factory
            HttpsURLConnection.setDefaultSSLSocketFactory(sslContext.getSocketFactory());
            
            // Set hostname verifier that accepts all hostnames (NOT for production!)
            HttpsURLConnection.setDefaultHostnameVerifier((hostname, session) -> true);
            
            System.out.println("SSL context configured to trust all certificates");
            
        } catch (Exception e) {
            System.out.println("Error configuring SSL context: " + e.getMessage());
        }
    }
    
    public static void main(String[] args) {
        SSLClient sslClient = new SSLClient();
        
        sslClient.simpleSSLConnection();
        
        System.out.println("\n" + "=".repeat(50) + "\n");
        
        // Uncomment the following line to trust all certificates (use with caution!)
        // sslClient.trustAllCertificates();
    }
}
```

## Summary

Java networking provides comprehensive capabilities for network programming:

### Socket Programming
- **TCP Sockets**: Reliable, connection-oriented communication
- **UDP Sockets**: Fast, connectionless communication
- **Multi-threading**: Handle multiple clients concurrently
- **NIO**: High-performance, scalable networking

### HTTP Communication
- **Modern HTTP Client (Java 11+)**: Asynchronous, HTTP/2 support
- **Legacy URLConnection**: Backward compatibility
- **SSL/TLS Support**: Secure communication

### Advanced Features
- **Non-blocking I/O**: Scalable server applications
- **SSL/TLS**: Encrypted communication
- **Network Utilities**: Interface information, DNS resolution
- **WebSocket**: Real-time bidirectional communication

### Best Practices
- Use appropriate protocol (TCP vs UDP) based on requirements
- Implement proper error handling and timeouts
- Use connection pooling for high-traffic applications
- Validate and sanitize all network input
- Implement proper SSL/TLS for secure communication
- Use asynchronous processing for better scalability
- Monitor and log network operations for debugging
- Consider security implications of network programming

### Common Use Cases
- Client-server applications
- Microservices communication
- Real-time chat applications
- File transfer systems
- Web services and APIs
- IoT device communication
- Distributed systems
