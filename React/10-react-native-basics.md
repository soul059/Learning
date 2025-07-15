# React Native Basics

## Table of Contents
- [What is React Native?](#what-is-react-native)
- [React Native vs React](#react-native-vs-react)
- [Getting Started](#getting-started)
- [Core Components](#core-components)
- [Styling in React Native](#styling-in-react-native)
- [Handling User Input](#handling-user-input)
- [Lists and ScrollViews](#lists-and-scrollviews)
- [Images and Media](#images-and-media)
- [Platform-Specific Code](#platform-specific-code)
- [Debugging](#debugging)

## What is React Native?

React Native is a framework for building native mobile applications using React. It allows you to write mobile apps using JavaScript and React while rendering to native platform components.

### Key Benefits:
- **Cross-Platform**: Write once, run on iOS and Android
- **Native Performance**: Apps compile to native code
- **React Knowledge**: Leverage existing React skills
- **Hot Reloading**: Fast development cycle
- **Large Community**: Extensive ecosystem and support

### How React Native Works:
1. **JavaScript Thread**: Runs your React Native code
2. **Native Thread**: Handles UI and native platform APIs
3. **Bridge**: Communicates between JavaScript and native threads
4. **Metro Bundler**: Bundles JavaScript code for the app

## React Native vs React

| Aspect | React (Web) | React Native |
|--------|-------------|--------------|
| **Rendering** | DOM elements | Native components |
| **Styling** | CSS | StyleSheet API |
| **Navigation** | Browser routing | Stack/Tab navigators |
| **Platform APIs** | Web APIs | Native platform APIs |
| **Build Output** | Web bundle | Native app (APK/IPA) |
| **Debugging** | Browser DevTools | React Native Debugger |

### Code Differences:
```javascript
// React (Web)
import React from 'react';

function App() {
  return (
    <div style={{ padding: 20 }}>
      <h1>Hello Web!</h1>
      <button onClick={() => alert('Clicked!')}>
        Click me
      </button>
    </div>
  );
}

// React Native
import React from 'react';
import { View, Text, TouchableOpacity, Alert, StyleSheet } from 'react-native';

function App() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Hello Mobile!</Text>
      <TouchableOpacity 
        style={styles.button}
        onPress={() => Alert.alert('Clicked!')}
      >
        <Text style={styles.buttonText}>Press me</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  button: {
    backgroundColor: '#007bff',
    padding: 12,
    borderRadius: 4,
  },
  buttonText: {
    color: 'white',
    textAlign: 'center',
  },
});
```

## Getting Started

### Prerequisites
```bash
# Install Node.js (v14 or newer)
# Install npm or yarn
# Install Git

# For iOS development (macOS only):
# Install Xcode from App Store
# Install Xcode Command Line Tools
xcode-select --install

# For Android development:
# Install Android Studio
# Set up Android SDK and emulator
```

### Using Expo (Recommended for Beginners)
```bash
# Install Expo CLI
npm install -g @expo/cli

# Create new project
npx create-expo-app MyFirstApp

# Navigate to project
cd MyFirstApp

# Start development server
npx expo start

# Run on device/simulator
npx expo start --ios     # iOS simulator
npx expo start --android # Android emulator
npx expo start --web     # Web browser
```

### Using React Native CLI (Bare Workflow)
```bash
# Install React Native CLI
npm install -g @react-native-community/cli

# Create new project
npx react-native init MyApp

# Navigate to project
cd MyApp

# For iOS (macOS only)
cd ios && pod install && cd ..
npx react-native run-ios

# For Android
npx react-native run-android
```

### Project Structure (Expo)
```
MyFirstApp/
├── app.json              # Expo configuration
├── App.js               # Main app component
├── package.json         # Dependencies
├── babel.config.js      # Babel configuration
├── assets/              # Images, fonts, etc.
│   ├── images/
│   └── fonts/
├── components/          # Reusable components
├── screens/            # Screen components
├── navigation/         # Navigation setup
├── services/           # API calls, utilities
└── styles/             # Style definitions
```

### Basic App.js Structure
```javascript
import React from 'react';
import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View } from 'react-native';

export default function App() {
  return (
    <View style={styles.container}>
      <Text style={styles.text}>Hello, React Native!</Text>
      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  text: {
    fontSize: 18,
    fontWeight: 'bold',
  },
});
```

## Core Components

### View Component
The building block of React Native UIs, similar to `<div>` in web.

```javascript
import React from 'react';
import { View, StyleSheet } from 'react-native';

function MyComponent() {
  return (
    <View style={styles.container}>
      <View style={styles.header}>
        {/* Header content */}
      </View>
      
      <View style={styles.content}>
        {/* Main content */}
      </View>
      
      <View style={styles.footer}>
        {/* Footer content */}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    height: 80,
    backgroundColor: '#007bff',
    justifyContent: 'center',
    alignItems: 'center',
  },
  content: {
    flex: 1,
    padding: 20,
  },
  footer: {
    height: 60,
    backgroundColor: '#6c757d',
    justifyContent: 'center',
    alignItems: 'center',
  },
});
```

### Text Component
For displaying text content.

```javascript
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

function TextExamples() {
  return (
    <View style={styles.container}>
      {/* Basic text */}
      <Text style={styles.title}>Welcome to React Native</Text>
      
      {/* Nested text with different styles */}
      <Text style={styles.paragraph}>
        This is a paragraph with {' '}
        <Text style={styles.bold}>bold</Text> and {' '}
        <Text style={styles.italic}>italic</Text> text.
      </Text>
      
      {/* Pressable text */}
      <Text 
        style={styles.link}
        onPress={() => console.log('Link pressed')}
      >
        This is a pressable link
      </Text>
      
      {/* Text with numberOfLines prop */}
      <Text 
        style={styles.description}
        numberOfLines={2}
        ellipsizeMode="tail"
      >
        This is a very long text that will be truncated after two lines. 
        The rest of the content will be hidden with an ellipsis.
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 16,
    textAlign: 'center',
  },
  paragraph: {
    fontSize: 16,
    lineHeight: 24,
    marginBottom: 16,
  },
  bold: {
    fontWeight: 'bold',
  },
  italic: {
    fontStyle: 'italic',
  },
  link: {
    fontSize: 16,
    color: '#007bff',
    textDecorationLine: 'underline',
    marginBottom: 16,
  },
  description: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
  },
});
```

### TouchableOpacity & Button
For handling user interactions.

```javascript
import React, { useState } from 'react';
import { 
  View, 
  Text, 
  TouchableOpacity, 
  TouchableHighlight,
  TouchableWithoutFeedback,
  Pressable,
  Button,
  Alert,
  StyleSheet 
} from 'react-native';

function TouchableExamples() {
  const [count, setCount] = useState(0);

  const showAlert = () => {
    Alert.alert(
      'Alert Title',
      'Alert message here',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'OK', onPress: () => console.log('OK pressed') }
      ]
    );
  };

  return (
    <View style={styles.container}>
      {/* TouchableOpacity - most commonly used */}
      <TouchableOpacity 
        style={styles.button}
        activeOpacity={0.7}
        onPress={() => setCount(count + 1)}
      >
        <Text style={styles.buttonText}>
          TouchableOpacity (Count: {count})
        </Text>
      </TouchableOpacity>

      {/* TouchableHighlight */}
      <TouchableHighlight
        style={styles.button}
        underlayColor="#0056b3"
        onPress={showAlert}
      >
        <Text style={styles.buttonText}>TouchableHighlight</Text>
      </TouchableHighlight>

      {/* Pressable (newer, more flexible) */}
      <Pressable
        style={({ pressed }) => [
          styles.button,
          pressed && styles.buttonPressed
        ]}
        onPress={() => console.log('Pressable pressed')}
        onLongPress={() => console.log('Long press detected')}
      >
        <Text style={styles.buttonText}>Pressable</Text>
      </Pressable>

      {/* Built-in Button component */}
      <View style={styles.buttonContainer}>
        <Button
          title="Built-in Button"
          onPress={() => console.log('Button pressed')}
          color="#28a745"
        />
      </View>

      {/* Disabled button */}
      <TouchableOpacity 
        style={[styles.button, styles.buttonDisabled]}
        disabled={true}
        onPress={() => console.log('This will not fire')}
      >
        <Text style={[styles.buttonText, styles.buttonTextDisabled]}>
          Disabled Button
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 20,
    gap: 16,
  },
  button: {
    backgroundColor: '#007bff',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  buttonPressed: {
    backgroundColor: '#0056b3',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  buttonContainer: {
    marginVertical: 8,
  },
  buttonDisabled: {
    backgroundColor: '#ccc',
  },
  buttonTextDisabled: {
    color: '#999',
  },
});
```

### SafeAreaView
Ensures content renders within safe area boundaries.

```javascript
import React from 'react';
import { SafeAreaView, View, Text, StyleSheet, StatusBar } from 'react-native';

function SafeAreaExample() {
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#f8f9fa" />
      
      <View style={styles.header}>
        <Text style={styles.headerText}>Header</Text>
      </View>
      
      <View style={styles.content}>
        <Text>Content is safe from notches and status bars!</Text>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
  },
  header: {
    height: 60,
    backgroundColor: '#007bff',
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
});
```

## Styling in React Native

### StyleSheet API
```javascript
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

function StylingExample() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Styling in React Native</Text>
      
      {/* Multiple styles */}
      <View style={[styles.box, styles.blueBox]}>
        <Text style={styles.boxText}>Blue Box</Text>
      </View>
      
      {/* Conditional styling */}
      <View style={[
        styles.box, 
        { backgroundColor: Math.random() > 0.5 ? 'red' : 'green' }
      ]}>
        <Text style={styles.boxText}>Random Color Box</Text>
      </View>
      
      {/* Inline styles (not recommended for performance) */}
      <Text style={{ fontSize: 16, color: 'purple', marginTop: 10 }}>
        Inline styled text
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
  },
  box: {
    width: 100,
    height: 100,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
    borderRadius: 8,
  },
  blueBox: {
    backgroundColor: '#007bff',
  },
  boxText: {
    color: 'white',
    fontWeight: 'bold',
  },
});
```

### Flexbox Layout
```javascript
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

function FlexboxExample() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Flexbox Layout</Text>
      
      {/* Row layout */}
      <View style={styles.row}>
        <View style={[styles.box, styles.box1]}>
          <Text style={styles.boxText}>1</Text>
        </View>
        <View style={[styles.box, styles.box2]}>
          <Text style={styles.boxText}>2</Text>
        </View>
        <View style={[styles.box, styles.box3]}>
          <Text style={styles.boxText}>3</Text>
        </View>
      </View>
      
      {/* Column layout with flex */}
      <View style={styles.column}>
        <View style={[styles.flexBox, { flex: 1, backgroundColor: '#ff6b6b' }]}>
          <Text style={styles.boxText}>Flex: 1</Text>
        </View>
        <View style={[styles.flexBox, { flex: 2, backgroundColor: '#4ecdc4' }]}>
          <Text style={styles.boxText}>Flex: 2</Text>
        </View>
        <View style={[styles.flexBox, { flex: 1, backgroundColor: '#45b7d1' }]}>
          <Text style={styles.boxText}>Flex: 1</Text>
        </View>
      </View>
      
      {/* Alignment examples */}
      <View style={styles.alignmentContainer}>
        <View style={styles.centered}>
          <Text style={styles.boxText}>Centered</Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  column: {
    height: 200,
    marginBottom: 20,
  },
  box: {
    width: 80,
    height: 80,
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 8,
  },
  box1: {
    backgroundColor: '#ff6b6b',
  },
  box2: {
    backgroundColor: '#4ecdc4',
  },
  box3: {
    backgroundColor: '#45b7d1',
  },
  flexBox: {
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 8,
    borderRadius: 8,
  },
  boxText: {
    color: 'white',
    fontWeight: 'bold',
    textAlign: 'center',
  },
  alignmentContainer: {
    height: 100,
    backgroundColor: '#e9ecef',
    borderRadius: 8,
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#6f42c1',
    margin: 20,
    borderRadius: 8,
  },
});
```

### Responsive Design
```javascript
import React from 'react';
import { View, Text, StyleSheet, Dimensions } from 'react-native';

const { width, height } = Dimensions.get('window');

function ResponsiveExample() {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Responsive Design</Text>
      
      <View style={styles.responsiveBox}>
        <Text style={styles.boxText}>
          Screen: {width.toFixed(0)}x{height.toFixed(0)}
        </Text>
      </View>
      
      <View style={styles.grid}>
        {[1, 2, 3, 4].map(num => (
          <View key={num} style={styles.gridItem}>
            <Text style={styles.boxText}>{num}</Text>
          </View>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
  },
  responsiveBox: {
    width: width * 0.8, // 80% of screen width
    height: height * 0.2, // 20% of screen height
    backgroundColor: '#007bff',
    justifyContent: 'center',
    alignItems: 'center',
    alignSelf: 'center',
    borderRadius: 8,
    marginBottom: 20,
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  gridItem: {
    width: (width - 60) / 2, // 2 columns with padding
    height: 80,
    backgroundColor: '#28a745',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 16,
    borderRadius: 8,
  },
  boxText: {
    color: 'white',
    fontWeight: 'bold',
    textAlign: 'center',
  },
});
```

## Handling User Input

### TextInput Component
```javascript
import React, { useState } from 'react';
import { 
  View, 
  Text, 
  TextInput, 
  TouchableOpacity, 
  Alert,
  StyleSheet 
} from 'react-native';

function InputExample() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [message, setMessage] = useState('');

  const handleSubmit = () => {
    if (!name || !email || !password) {
      Alert.alert('Error', 'Please fill in all fields');
      return;
    }
    
    Alert.alert('Success', `Hello ${name}!`);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>User Input Form</Text>
      
      {/* Basic text input */}
      <TextInput
        style={styles.input}
        placeholder="Enter your name"
        value={name}
        onChangeText={setName}
        autoCapitalize="words"
        autoCorrect={false}
      />
      
      {/* Email input */}
      <TextInput
        style={styles.input}
        placeholder="Enter your email"
        value={email}
        onChangeText={setEmail}
        keyboardType="email-address"
        autoCapitalize="none"
        autoCorrect={false}
      />
      
      {/* Password input */}
      <TextInput
        style={styles.input}
        placeholder="Enter your password"
        value={password}
        onChangeText={setPassword}
        secureTextEntry={true}
        autoCapitalize="none"
      />
      
      {/* Multiline text input */}
      <TextInput
        style={[styles.input, styles.textArea]}
        placeholder="Enter your message"
        value={message}
        onChangeText={setMessage}
        multiline={true}
        numberOfLines={4}
        textAlignVertical="top"
      />
      
      <TouchableOpacity style={styles.button} onPress={handleSubmit}>
        <Text style={styles.buttonText}>Submit</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 30,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    padding: 16,
    borderRadius: 8,
    backgroundColor: 'white',
    marginBottom: 16,
    fontSize: 16,
  },
  textArea: {
    height: 100,
    textAlignVertical: 'top',
  },
  button: {
    backgroundColor: '#007bff',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 20,
  },
  buttonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
});
```

### Form Validation
```javascript
import React, { useState } from 'react';
import { 
  View, 
  Text, 
  TextInput, 
  TouchableOpacity, 
  StyleSheet 
} from 'react-native';

function ValidatedForm() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    phone: '',
  });
  const [errors, setErrors] = useState({});

  const validateEmail = (email) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const validatePhone = (phone) => {
    const phoneRegex = /^\+?[\d\s-()]{10,}$/;
    return phoneRegex.test(phone);
  };

  const handleInputChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    
    // Clear error when user starts typing
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }));
    }
  };

  const validateForm = () => {
    const newErrors = {};

    if (!formData.name.trim()) {
      newErrors.name = 'Name is required';
    }

    if (!formData.email.trim()) {
      newErrors.email = 'Email is required';
    } else if (!validateEmail(formData.email)) {
      newErrors.email = 'Please enter a valid email';
    }

    if (!formData.phone.trim()) {
      newErrors.phone = 'Phone number is required';
    } else if (!validatePhone(formData.phone)) {
      newErrors.phone = 'Please enter a valid phone number';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = () => {
    if (validateForm()) {
      console.log('Form submitted:', formData);
      // Reset form
      setFormData({ name: '', email: '', phone: '' });
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Validated Form</Text>
      
      <View style={styles.inputContainer}>
        <TextInput
          style={[styles.input, errors.name && styles.inputError]}
          placeholder="Full Name"
          value={formData.name}
          onChangeText={(value) => handleInputChange('name', value)}
        />
        {errors.name && <Text style={styles.errorText}>{errors.name}</Text>}
      </View>
      
      <View style={styles.inputContainer}>
        <TextInput
          style={[styles.input, errors.email && styles.inputError]}
          placeholder="Email Address"
          value={formData.email}
          onChangeText={(value) => handleInputChange('email', value)}
          keyboardType="email-address"
          autoCapitalize="none"
        />
        {errors.email && <Text style={styles.errorText}>{errors.email}</Text>}
      </View>
      
      <View style={styles.inputContainer}>
        <TextInput
          style={[styles.input, errors.phone && styles.inputError]}
          placeholder="Phone Number"
          value={formData.phone}
          onChangeText={(value) => handleInputChange('phone', value)}
          keyboardType="phone-pad"
        />
        {errors.phone && <Text style={styles.errorText}>{errors.phone}</Text>}
      </View>
      
      <TouchableOpacity style={styles.button} onPress={handleSubmit}>
        <Text style={styles.buttonText}>Submit</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 30,
  },
  inputContainer: {
    marginBottom: 20,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    padding: 16,
    borderRadius: 8,
    backgroundColor: 'white',
    fontSize: 16,
  },
  inputError: {
    borderColor: '#dc3545',
  },
  errorText: {
    color: '#dc3545',
    fontSize: 14,
    marginTop: 4,
    marginLeft: 4,
  },
  button: {
    backgroundColor: '#007bff',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 20,
  },
  buttonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
});
```

## Lists and ScrollViews

### ScrollView
```javascript
import React from 'react';
import { 
  ScrollView, 
  View, 
  Text, 
  TouchableOpacity, 
  StyleSheet 
} from 'react-native';

function ScrollViewExample() {
  const items = Array.from({ length: 20 }, (_, i) => `Item ${i + 1}`);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>ScrollView Example</Text>
      
      <ScrollView 
        style={styles.scrollView}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
      >
        {items.map((item, index) => (
          <TouchableOpacity 
            key={index} 
            style={styles.item}
            onPress={() => console.log(`Pressed ${item}`)}
          >
            <Text style={styles.itemText}>{item}</Text>
          </TouchableOpacity>
        ))}
      </ScrollView>
      
      {/* Horizontal ScrollView */}
      <Text style={styles.subtitle}>Horizontal Scroll</Text>
      <ScrollView 
        horizontal 
        showsHorizontalScrollIndicator={false}
        style={styles.horizontalScroll}
      >
        {items.slice(0, 10).map((item, index) => (
          <View key={index} style={styles.horizontalItem}>
            <Text style={styles.horizontalItemText}>{item}</Text>
          </View>
        ))}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    padding: 20,
  },
  subtitle: {
    fontSize: 18,
    fontWeight: 'bold',
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 10,
  },
  scrollView: {
    flex: 1,
    paddingHorizontal: 20,
  },
  scrollContent: {
    paddingBottom: 20,
  },
  item: {
    backgroundColor: 'white',
    padding: 20,
    marginBottom: 10,
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  itemText: {
    fontSize: 16,
    textAlign: 'center',
  },
  horizontalScroll: {
    paddingHorizontal: 20,
  },
  horizontalItem: {
    backgroundColor: '#007bff',
    padding: 20,
    marginRight: 10,
    borderRadius: 8,
    minWidth: 120,
    justifyContent: 'center',
    alignItems: 'center',
  },
  horizontalItemText: {
    color: 'white',
    fontWeight: 'bold',
  },
});
```

### FlatList
```javascript
import React, { useState } from 'react';
import { 
  View, 
  Text, 
  FlatList, 
  TouchableOpacity, 
  ActivityIndicator,
  RefreshControl,
  StyleSheet 
} from 'react-native';

function FlatListExample() {
  const [data, setData] = useState(
    Array.from({ length: 50 }, (_, i) => ({
      id: i.toString(),
      title: `Item ${i + 1}`,
      description: `This is the description for item ${i + 1}`,
    }))
  );
  const [refreshing, setRefreshing] = useState(false);
  const [loading, setLoading] = useState(false);

  const onRefresh = () => {
    setRefreshing(true);
    // Simulate API call
    setTimeout(() => {
      setData(prevData => 
        prevData.map(item => ({
          ...item,
          title: `${item.title} (refreshed)`,
        }))
      );
      setRefreshing(false);
    }, 2000);
  };

  const loadMore = () => {
    if (loading) return;
    
    setLoading(true);
    // Simulate loading more data
    setTimeout(() => {
      const newData = Array.from({ length: 10 }, (_, i) => ({
        id: (data.length + i).toString(),
        title: `Item ${data.length + i + 1}`,
        description: `This is the description for item ${data.length + i + 1}`,
      }));
      setData(prevData => [...prevData, ...newData]);
      setLoading(false);
    }, 1500);
  };

  const renderItem = ({ item, index }) => (
    <TouchableOpacity 
      style={styles.item}
      onPress={() => console.log(`Pressed item ${item.id}`)}
    >
      <Text style={styles.itemTitle}>{item.title}</Text>
      <Text style={styles.itemDescription}>{item.description}</Text>
    </TouchableOpacity>
  );

  const renderFooter = () => {
    if (!loading) return null;
    
    return (
      <View style={styles.footer}>
        <ActivityIndicator size="large" color="#007bff" />
        <Text style={styles.footerText}>Loading more...</Text>
      </View>
    );
  };

  const renderSeparator = () => (
    <View style={styles.separator} />
  );

  return (
    <View style={styles.container}>
      <Text style={styles.title}>FlatList Example</Text>
      
      <FlatList
        data={data}
        renderItem={renderItem}
        keyExtractor={item => item.id}
        ItemSeparatorComponent={renderSeparator}
        ListFooterComponent={renderFooter}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            colors={['#007bff']}
          />
        }
        onEndReached={loadMore}
        onEndReachedThreshold={0.1}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.listContent}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    padding: 20,
  },
  listContent: {
    paddingHorizontal: 20,
  },
  item: {
    backgroundColor: 'white',
    padding: 20,
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.1,
    shadowRadius: 3.84,
    elevation: 5,
  },
  itemTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  itemDescription: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
  },
  separator: {
    height: 12,
  },
  footer: {
    paddingVertical: 20,
    alignItems: 'center',
  },
  footerText: {
    fontSize: 16,
    color: '#666',
    marginTop: 10,
  },
});
```

## Images and Media

### Image Component
```javascript
import React from 'react';
import { 
  View, 
  Text, 
  Image, 
  ScrollView, 
  StyleSheet,
  Dimensions 
} from 'react-native';

const { width } = Dimensions.get('window');

function ImageExample() {
  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Image Examples</Text>
      
      {/* Local image */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Local Image</Text>
        <Image 
          source={require('../assets/react-native-logo.png')} 
          style={styles.localImage}
          resizeMode="contain"
        />
      </View>
      
      {/* Network image */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Network Image</Text>
        <Image 
          source={{ uri: 'https://picsum.photos/300/200' }}
          style={styles.networkImage}
          resizeMode="cover"
        />
      </View>
      
      {/* Image with different resize modes */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Resize Modes</Text>
        
        {['cover', 'contain', 'stretch', 'repeat', 'center'].map(mode => (
          <View key={mode} style={styles.resizeModeContainer}>
            <Text style={styles.resizeModeText}>{mode}</Text>
            <Image 
              source={{ uri: 'https://picsum.photos/200/100' }}
              style={styles.resizeModeImage}
              resizeMode={mode}
            />
          </View>
        ))}
      </View>
      
      {/* Rounded images */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Rounded Images</Text>
        
        <View style={styles.roundedContainer}>
          <Image 
            source={{ uri: 'https://picsum.photos/100/100' }}
            style={styles.circularImage}
          />
          
          <Image 
            source={{ uri: 'https://picsum.photos/100/100' }}
            style={styles.roundedRectImage}
          />
        </View>
      </View>
      
      {/* Image with loading and error handling */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Image with Fallback</Text>
        <ImageWithFallback 
          source={{ uri: 'https://invalid-url.com/image.jpg' }}
          fallbackSource={{ uri: 'https://picsum.photos/300/200' }}
          style={styles.networkImage}
        />
      </View>
    </ScrollView>
  );
}

// Custom component with fallback handling
function ImageWithFallback({ source, fallbackSource, style, ...props }) {
  const [imageSource, setImageSource] = React.useState(source);
  const [isLoading, setIsLoading] = React.useState(true);

  return (
    <View style={style}>
      <Image
        source={imageSource}
        style={StyleSheet.absoluteFillObject}
        onLoad={() => setIsLoading(false)}
        onError={() => {
          setImageSource(fallbackSource);
          setIsLoading(false);
        }}
        {...props}
      />
      
      {isLoading && (
        <View style={[StyleSheet.absoluteFillObject, styles.loadingContainer]}>
          <Text style={styles.loadingText}>Loading...</Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    padding: 20,
  },
  section: {
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  localImage: {
    width: 200,
    height: 100,
    alignSelf: 'center',
  },
  networkImage: {
    width: width - 40,
    height: 200,
    borderRadius: 8,
  },
  resizeModeContainer: {
    marginBottom: 16,
  },
  resizeModeText: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  resizeModeImage: {
    width: 150,
    height: 100,
    backgroundColor: '#e9ecef',
    borderWidth: 1,
    borderColor: '#ddd',
  },
  roundedContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
  },
  circularImage: {
    width: 80,
    height: 80,
    borderRadius: 40,
  },
  roundedRectImage: {
    width: 100,
    height: 80,
    borderRadius: 16,
  },
  loadingContainer: {
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f8f9fa',
  },
  loadingText: {
    color: '#666',
    fontSize: 16,
  },
});
```

## Platform-Specific Code

### Platform Module
```javascript
import React from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  Platform,
  StatusBar 
} from 'react-native';

function PlatformExample() {
  return (
    <View style={styles.container}>
      <StatusBar 
        barStyle={Platform.OS === 'ios' ? 'dark-content' : 'light-content'}
        backgroundColor={Platform.OS === 'android' ? '#007bff' : undefined}
      />
      
      <Text style={styles.title}>Platform-Specific Code</Text>
      
      <View style={styles.infoContainer}>
        <Text style={styles.infoText}>
          Current Platform: {Platform.OS}
        </Text>
        <Text style={styles.infoText}>
          Platform Version: {Platform.Version}
        </Text>
        {Platform.OS === 'ios' && (
          <Text style={styles.infoText}>
            Is iPad: {Platform.isPad ? 'Yes' : 'No'}
          </Text>
        )}
      </View>
      
      <View style={styles.platformBox}>
        <Text style={styles.boxText}>
          This box has platform-specific styling
        </Text>
      </View>
      
      {/* Conditional rendering based on platform */}
      {Platform.OS === 'ios' ? (
        <Text style={styles.platformText}>iOS specific content</Text>
      ) : (
        <Text style={styles.platformText}>Android specific content</Text>
      )}
      
      {/* Platform.select usage */}
      <Text style={[styles.selectText, platformSelectStyle]}>
        Platform.select styling
      </Text>
    </View>
  );
}

// Platform.select for styles
const platformSelectStyle = Platform.select({
  ios: {
    fontFamily: 'Helvetica',
    fontSize: 18,
  },
  android: {
    fontFamily: 'Roboto',
    fontSize: 16,
  },
  default: {
    fontFamily: 'Arial',
    fontSize: 17,
  },
});

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#f5f5f5',
    paddingTop: Platform.OS === 'android' ? StatusBar.currentHeight + 20 : 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 30,
  },
  infoContainer: {
    backgroundColor: 'white',
    padding: 20,
    borderRadius: 8,
    marginBottom: 20,
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: {
          width: 0,
          height: 2,
        },
        shadowOpacity: 0.25,
        shadowRadius: 3.84,
      },
      android: {
        elevation: 5,
      },
    }),
  },
  infoText: {
    fontSize: 16,
    marginBottom: 8,
  },
  platformBox: {
    padding: 20,
    borderRadius: 8,
    marginBottom: 20,
    ...Platform.select({
      ios: {
        backgroundColor: '#007AFF', // iOS blue
      },
      android: {
        backgroundColor: '#4CAF50', // Material green
      },
      default: {
        backgroundColor: '#666',
      },
    }),
  },
  boxText: {
    color: 'white',
    textAlign: 'center',
    fontSize: 16,
    fontWeight: 'bold',
  },
  platformText: {
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 20,
    fontStyle: 'italic',
  },
  selectText: {
    textAlign: 'center',
    color: '#007bff',
    fontWeight: 'bold',
  },
});
```

### Platform-Specific Files
```javascript
// Button.ios.js
import React from 'react';
import { TouchableOpacity, Text, StyleSheet } from 'react-native';

export default function Button({ title, onPress }) {
  return (
    <TouchableOpacity style={styles.button} onPress={onPress}>
      <Text style={styles.text}>{title}</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  button: {
    backgroundColor: '#007AFF',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  text: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
});

// Button.android.js
import React from 'react';
import { TouchableNativeFeedback, View, Text, StyleSheet } from 'react-native';

export default function Button({ title, onPress }) {
  return (
    <TouchableNativeFeedback 
      onPress={onPress}
      background={TouchableNativeFeedback.Ripple('#ffffff40', false)}
    >
      <View style={styles.button}>
        <Text style={styles.text}>{title}</Text>
      </View>
    </TouchableNativeFeedback>
  );
}

const styles = StyleSheet.create({
  button: {
    backgroundColor: '#2196F3',
    padding: 16,
    borderRadius: 4,
    alignItems: 'center',
    elevation: 2,
  },
  text: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
    textTransform: 'uppercase',
  },
});

// Usage (React Native will automatically pick the right file)
import Button from './components/Button';

function App() {
  return (
    <Button 
      title="Platform Button" 
      onPress={() => console.log('Pressed')} 
    />
  );
}
```

## Debugging

### Debug Techniques
```javascript
import React, { useEffect } from 'react';
import { View, Text, Button, StyleSheet } from 'react-native';

function DebugExample() {
  useEffect(() => {
    // Console logging
    console.log('Component mounted');
    console.warn('This is a warning');
    console.error('This is an error');
    
    // Object logging
    const user = { id: 1, name: 'John Doe' };
    console.log('User object:', user);
    
    return () => {
      console.log('Component unmounted');
    };
  }, []);

  const handleDebug = () => {
    // Debugger statement (works with remote debugging)
    debugger;
    
    // Manual breakpoint alternative
    console.log('Debug point reached');
    
    // Inspect variables
    const data = { timestamp: new Date(), random: Math.random() };
    console.table(data); // Works in some debuggers
  };

  const handleError = () => {
    try {
      throw new Error('Intentional error for testing');
    } catch (error) {
      console.error('Caught error:', error.message);
      console.error('Stack trace:', error.stack);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Debug Example</Text>
      
      <Button title="Trigger Debug Point" onPress={handleDebug} />
      <Button title="Trigger Error" onPress={handleError} />
      
      {/* Conditional rendering for debugging */}
      {__DEV__ && (
        <View style={styles.debugInfo}>
          <Text style={styles.debugText}>Development Mode</Text>
          <Text style={styles.debugText}>
            Timestamp: {new Date().toLocaleTimeString()}
          </Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    justifyContent: 'center',
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 30,
  },
  debugInfo: {
    marginTop: 30,
    padding: 16,
    backgroundColor: '#fff3cd',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#ffeaa7',
  },
  debugText: {
    fontSize: 14,
    color: '#856404',
    textAlign: 'center',
  },
});
```

### Performance Debugging
```javascript
import React, { useState, useCallback } from 'react';
import { 
  View, 
  Text, 
  FlatList, 
  TouchableOpacity, 
  StyleSheet 
} from 'react-native';

// Performance monitoring
const performanceLog = (label) => {
  const start = Date.now();
  return () => {
    const end = Date.now();
    console.log(`${label} took ${end - start}ms`);
  };
};

function PerformanceExample() {
  const [data, setData] = useState(
    Array.from({ length: 1000 }, (_, i) => ({ id: i, title: `Item ${i}` }))
  );

  const renderItem = useCallback(({ item }) => {
    const endLog = performanceLog(`Render item ${item.id}`);
    
    // Simulate expensive operation
    const result = (
      <TouchableOpacity 
        style={styles.item}
        onPress={() => console.log(`Pressed ${item.id}`)}
      >
        <Text>{item.title}</Text>
      </TouchableOpacity>
    );
    
    endLog();
    return result;
  }, []);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Performance Debugging</Text>
      
      <FlatList
        data={data}
        renderItem={renderItem}
        keyExtractor={item => item.id.toString()}
        initialNumToRender={10}
        maxToRenderPerBatch={10}
        windowSize={10}
        removeClippedSubviews={true}
        getItemLayout={(data, index) => ({
          length: 50,
          offset: 50 * index,
          index,
        })}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    textAlign: 'center',
    padding: 20,
  },
  item: {
    padding: 16,
    backgroundColor: 'white',
    marginVertical: 1,
  },
});
```

### Debug Menu and Tools
```javascript
// Access debug menu:
// iOS Simulator: Cmd + D
// Android Emulator: Cmd + M (Mac) or Ctrl + M (Windows/Linux)
// Physical device: Shake the device

// Enable remote debugging in debug menu
// Then open Chrome DevTools at chrome://inspect

// React Native Debugger (standalone app)
// Download from: https://github.com/jhen0409/react-native-debugger

// Flipper (Facebook's debugging platform)
// Automatically enabled in new React Native projects
// Provides network inspector, layout inspector, and more

import { NativeModules } from 'react-native';

function DevMenuExample() {
  const openDevMenu = () => {
    if (__DEV__) {
      NativeModules.DevMenu?.show();
    }
  };

  return (
    <View style={styles.container}>
      <Text>Debug Tools Available:</Text>
      <Text>• Remote JS Debugging</Text>
      <Text>• Element Inspector</Text>
      <Text>• Performance Monitor</Text>
      <Text>• Network Inspector</Text>
      <Text>• React DevTools</Text>
      
      {__DEV__ && (
        <TouchableOpacity onPress={openDevMenu} style={styles.button}>
          <Text style={styles.buttonText}>Open Dev Menu</Text>
        </TouchableOpacity>
      )}
    </View>
  );
}
```

---

*Continue to: [11-react-native-components.md](./11-react-native-components.md)*
