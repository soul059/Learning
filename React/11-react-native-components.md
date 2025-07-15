# React Native Components and Styling

## Table of Contents
- [Core Components Deep Dive](#core-components-deep-dive)
- [Advanced Styling](#advanced-styling)
- [Custom Components](#custom-components)
- [Layout Systems](#layout-systems)
- [Animations](#animations)
- [Gestures](#gestures)
- [Performance Optimization](#performance-optimization)
- [Platform-Specific Components](#platform-specific-components)

## Core Components Deep Dive

### Advanced Text Component
```javascript
import React from 'react';
import { Text, StyleSheet, Linking } from 'react-native';

function AdvancedTextExample() {
  const handleLinkPress = (url) => {
    Linking.openURL(url);
  };

  return (
    <Text style={styles.container}>
      <Text style={styles.title}>Advanced Text Features</Text>
      {'\n\n'}
      
      {/* Nested text with different styles */}
      <Text style={styles.paragraph}>
        This is a paragraph with{' '}
        <Text style={styles.bold}>bold text</Text>,{' '}
        <Text style={styles.italic}>italic text</Text>, and{' '}
        <Text 
          style={styles.link}
          onPress={() => handleLinkPress('https://reactnative.dev')}
        >
          clickable links
        </Text>.
      </Text>
      {'\n\n'}
      
      {/* Selectable text */}
      <Text selectable style={styles.selectable}>
        This text can be selected and copied by the user.
      </Text>
      {'\n\n'}
      
      {/* Text with shadow */}
      <Text style={styles.shadowText}>Text with shadow</Text>
      {'\n\n'}
      
      {/* Adjustable font size */}
      <Text 
        style={styles.adjustable}
        adjustsFontSizeToFit={true}
        numberOfLines={1}
        minimumFontScale={0.5}
      >
        This text will adjust its size to fit in one line
      </Text>
    </Text>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 20,
    fontSize: 16,
    lineHeight: 24,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    color: '#333',
  },
  paragraph: {
    fontSize: 16,
    lineHeight: 24,
    textAlign: 'justify',
  },
  bold: {
    fontWeight: 'bold',
  },
  italic: {
    fontStyle: 'italic',
  },
  link: {
    color: '#007bff',
    textDecorationLine: 'underline',
  },
  selectable: {
    backgroundColor: '#f8f9fa',
    padding: 10,
    borderRadius: 5,
    fontFamily: 'monospace',
  },
  shadowText: {
    fontSize: 20,
    fontWeight: 'bold',
    textShadowColor: 'rgba(0, 0, 0, 0.3)',
    textShadowOffset: { width: 2, height: 2 },
    textShadowRadius: 4,
  },
  adjustable: {
    fontSize: 18,
    backgroundColor: '#e9ecef',
    padding: 10,
    borderRadius: 5,
  },
});
```

### Advanced Image Component
```javascript
import React, { useState } from 'react';
import { 
  View, 
  Image, 
  Text, 
  StyleSheet, 
  ActivityIndicator,
  TouchableOpacity 
} from 'react-native';

function AdvancedImageExample() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [imageSize, setImageSize] = useState(null);

  const handleImageLoad = (event) => {
    const { width, height } = event.nativeEvent.source;
    setImageSize({ width, height });
    setLoading(false);
  };

  const handleImageError = () => {
    setError(true);
    setLoading(false);
  };

  const retryImage = () => {
    setError(false);
    setLoading(true);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Advanced Image Features</Text>
      
      {/* Progressive loading with placeholder */}
      <View style={styles.imageContainer}>
        {loading && (
          <View style={styles.loadingOverlay}>
            <ActivityIndicator size="large" color="#007bff" />
            <Text>Loading image...</Text>
          </View>
        )}
        
        {error ? (
          <View style={styles.errorContainer}>
            <Text style={styles.errorText}>Failed to load image</Text>
            <TouchableOpacity onPress={retryImage} style={styles.retryButton}>
              <Text style={styles.retryText}>Retry</Text>
            </TouchableOpacity>
          </View>
        ) : (
          <Image
            source={{ uri: 'https://picsum.photos/300/200' }}
            style={styles.image}
            onLoad={handleImageLoad}
            onError={handleImageError}
            onLoadStart={() => setLoading(true)}
            resizeMode="cover"
            // Progressive JPEG loading
            progressiveRenderingEnabled={true}
            // Caching (iOS)
            defaultSource={require('../assets/placeholder.png')}
          />
        )}
      </View>
      
      {imageSize && (
        <Text style={styles.sizeText}>
          Image size: {imageSize.width}x{imageSize.height}
        </Text>
      )}
      
      {/* Different resize modes */}
      <View style={styles.resizeModeContainer}>
        <Text style={styles.subtitle}>Resize Modes:</Text>
        
        {['cover', 'contain', 'stretch', 'repeat', 'center'].map(mode => (
          <View key={mode} style={styles.resizeModeItem}>
            <Text style={styles.resizeModeLabel}>{mode}</Text>
            <Image
              source={{ uri: 'https://picsum.photos/100/80' }}
              style={styles.resizeModeImage}
              resizeMode={mode}
            />
          </View>
        ))}
      </View>
      
      {/* Background image */}
      <ImageBackground
        source={{ uri: 'https://picsum.photos/300/150' }}
        style={styles.backgroundImage}
        imageStyle={styles.backgroundImageStyle}
      >
        <Text style={styles.backgroundText}>Text over background image</Text>
      </ImageBackground>
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
    textAlign: 'center',
    marginBottom: 20,
  },
  subtitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginVertical: 10,
  },
  imageContainer: {
    height: 200,
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
    overflow: 'hidden',
    position: 'relative',
    marginBottom: 10,
  },
  image: {
    width: '100%',
    height: '100%',
  },
  loadingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(248, 249, 250, 0.8)',
    zIndex: 1,
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  errorText: {
    color: '#dc3545',
    marginBottom: 10,
  },
  retryButton: {
    backgroundColor: '#007bff',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 5,
  },
  retryText: {
    color: 'white',
    fontWeight: 'bold',
  },
  sizeText: {
    textAlign: 'center',
    color: '#666',
    marginBottom: 20,
  },
  resizeModeContainer: {
    marginVertical: 20,
  },
  resizeModeItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  resizeModeLabel: {
    width: 80,
    fontWeight: 'bold',
  },
  resizeModeImage: {
    width: 100,
    height: 60,
    backgroundColor: '#e9ecef',
    borderWidth: 1,
    borderColor: '#ddd',
  },
  backgroundImage: {
    height: 150,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 20,
  },
  backgroundImageStyle: {
    borderRadius: 8,
  },
  backgroundText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'white',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    padding: 10,
    borderRadius: 5,
  },
});
```

### Advanced ScrollView
```javascript
import React, { useState, useRef } from 'react';
import {
  ScrollView,
  View,
  Text,
  StyleSheet,
  Dimensions,
  TouchableOpacity,
} from 'react-native';

const { width } = Dimensions.get('window');

function AdvancedScrollViewExample() {
  const [scrollOffset, setScrollOffset] = useState(0);
  const scrollViewRef = useRef(null);

  const scrollToTop = () => {
    scrollViewRef.current?.scrollTo({ y: 0, animated: true });
  };

  const scrollToPosition = (y) => {
    scrollViewRef.current?.scrollTo({ y, animated: true });
  };

  const handleScroll = (event) => {
    const offsetY = event.nativeEvent.contentOffset.y;
    setScrollOffset(offsetY);
  };

  return (
    <View style={styles.container}>
      {/* Scroll indicator */}
      <View style={styles.header}>
        <Text>Scroll Offset: {Math.round(scrollOffset)}px</Text>
        <TouchableOpacity onPress={scrollToTop} style={styles.button}>
          <Text style={styles.buttonText}>Scroll to Top</Text>
        </TouchableOpacity>
      </View>

      <ScrollView
        ref={scrollViewRef}
        style={styles.scrollView}
        onScroll={handleScroll}
        scrollEventThrottle={16} // 60 FPS
        showsVerticalScrollIndicator={true}
        bounces={true}
        bouncesZoom={true}
        alwaysBounceVertical={false}
        decelerationRate="normal"
        snapToInterval={100} // Snap every 100px
        snapToAlignment="start"
        // Pull to refresh
        refreshControl={
          <RefreshControl
            refreshing={false}
            onRefresh={() => console.log('Refreshing...')}
            colors={['#007bff']}
            tintColor="#007bff"
          />
        }
      >
        {Array.from({ length: 50 }, (_, i) => (
          <View key={i} style={styles.item}>
            <Text style={styles.itemText}>Item {i + 1}</Text>
            <TouchableOpacity
              onPress={() => scrollToPosition((i + 10) * 100)}
              style={styles.jumpButton}
            >
              <Text style={styles.jumpButtonText}>Jump +10</Text>
            </TouchableOpacity>
          </View>
        ))}
      </ScrollView>

      {/* Horizontal ScrollView with paging */}
      <View style={styles.horizontalSection}>
        <Text style={styles.sectionTitle}>Horizontal Paging</Text>
        <ScrollView
          horizontal
          pagingEnabled
          showsHorizontalScrollIndicator={false}
          style={styles.horizontalScrollView}
        >
          {['Red', 'Green', 'Blue', 'Yellow', 'Purple'].map((color, index) => (
            <View
              key={index}
              style={[
                styles.page,
                { backgroundColor: color.toLowerCase() }
              ]}
            >
              <Text style={styles.pageText}>{color} Page</Text>
            </View>
          ))}
        </ScrollView>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#f8f9fa',
    borderBottomWidth: 1,
    borderBottomColor: '#dee2e6',
  },
  button: {
    backgroundColor: '#007bff',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 4,
  },
  buttonText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
  scrollView: {
    flex: 1,
  },
  item: {
    height: 100,
    backgroundColor: 'white',
    marginVertical: 1,
    paddingHorizontal: 20,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  itemText: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  jumpButton: {
    backgroundColor: '#28a745',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 4,
  },
  jumpButtonText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
  horizontalSection: {
    height: 200,
    backgroundColor: '#f8f9fa',
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    padding: 16,
  },
  horizontalScrollView: {
    flex: 1,
  },
  page: {
    width,
    justifyContent: 'center',
    alignItems: 'center',
  },
  pageText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
  },
});
```

## Advanced Styling

### Dynamic Styling
```javascript
import React, { useState } from 'react';
import { 
  View, 
  Text, 
  TouchableOpacity, 
  StyleSheet, 
  Dimensions,
  useColorScheme 
} from 'react-native';

const { width, height } = Dimensions.get('window');

function DynamicStylingExample() {
  const [theme, setTheme] = useState('light');
  const [size, setSize] = useState('medium');
  const [orientation, setOrientation] = useState('portrait');
  const colorScheme = useColorScheme();

  // Dynamic theme colors
  const colors = {
    light: {
      background: '#ffffff',
      text: '#333333',
      primary: '#007bff',
      secondary: '#6c757d',
    },
    dark: {
      background: '#121212',
      text: '#ffffff',
      primary: '#0d6efd',
      secondary: '#6c757d',
    },
  };

  // Size-based styles
  const sizes = {
    small: {
      fontSize: 14,
      padding: 8,
    },
    medium: {
      fontSize: 16,
      padding: 12,
    },
    large: {
      fontSize: 18,
      padding: 16,
    },
  };

  // Dynamic styles based on state
  const dynamicStyles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: colors[theme].background,
      padding: sizes[size].padding,
    },
    text: {
      color: colors[theme].text,
      fontSize: sizes[size].fontSize,
    },
    button: {
      backgroundColor: colors[theme].primary,
      padding: sizes[size].padding,
      borderRadius: 8,
      marginVertical: 4,
      alignItems: 'center',
    },
    buttonText: {
      color: 'white',
      fontSize: sizes[size].fontSize,
      fontWeight: 'bold',
    },
  });

  // Responsive styles
  const responsiveStyles = StyleSheet.create({
    responsiveContainer: {
      flexDirection: width > height ? 'row' : 'column',
      flex: 1,
    },
    responsiveBox: {
      flex: 1,
      backgroundColor: colors[theme].secondary,
      margin: 4,
      minHeight: width > height ? 100 : 150,
      justifyContent: 'center',
      alignItems: 'center',
    },
  });

  return (
    <View style={dynamicStyles.container}>
      <Text style={[dynamicStyles.text, styles.title]}>
        Dynamic Styling Example
      </Text>

      {/* Theme controls */}
      <View style={styles.controlGroup}>
        <Text style={dynamicStyles.text}>Theme:</Text>
        <View style={styles.buttonRow}>
          <TouchableOpacity
            style={[
              dynamicStyles.button,
              theme === 'light' && styles.activeButton
            ]}
            onPress={() => setTheme('light')}
          >
            <Text style={dynamicStyles.buttonText}>Light</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              dynamicStyles.button,
              theme === 'dark' && styles.activeButton
            ]}
            onPress={() => setTheme('dark')}
          >
            <Text style={dynamicStyles.buttonText}>Dark</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              dynamicStyles.button,
              theme === colorScheme && styles.activeButton
            ]}
            onPress={() => setTheme(colorScheme)}
          >
            <Text style={dynamicStyles.buttonText}>System</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Size controls */}
      <View style={styles.controlGroup}>
        <Text style={dynamicStyles.text}>Size:</Text>
        <View style={styles.buttonRow}>
          {['small', 'medium', 'large'].map(s => (
            <TouchableOpacity
              key={s}
              style={[
                dynamicStyles.button,
                size === s && styles.activeButton
              ]}
              onPress={() => setSize(s)}
            >
              <Text style={dynamicStyles.buttonText}>
                {s.charAt(0).toUpperCase() + s.slice(1)}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Responsive layout */}
      <View style={responsiveStyles.responsiveContainer}>
        <View style={responsiveStyles.responsiveBox}>
          <Text style={dynamicStyles.text}>Box 1</Text>
        </View>
        <View style={responsiveStyles.responsiveBox}>
          <Text style={dynamicStyles.text}>Box 2</Text>
        </View>
        <View style={responsiveStyles.responsiveBox}>
          <Text style={dynamicStyles.text}>Box 3</Text>
        </View>
      </View>

      <Text style={[dynamicStyles.text, styles.info]}>
        Screen: {Math.round(width)}x{Math.round(height)}
        {'\n'}
        Color Scheme: {colorScheme}
        {'\n'}
        Orientation: {width > height ? 'Landscape' : 'Portrait'}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
  },
  controlGroup: {
    marginBottom: 16,
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: 8,
  },
  activeButton: {
    opacity: 0.7,
    borderWidth: 2,
    borderColor: '#ffc107',
  },
  info: {
    textAlign: 'center',
    marginTop: 20,
    fontFamily: 'monospace',
  },
});
```

### Advanced Layout Techniques
```javascript
import React from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';

function AdvancedLayoutExample() {
  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Advanced Layout Techniques</Text>

      {/* Absolute positioning */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Absolute Positioning</Text>
        <View style={styles.relativeContainer}>
          <View style={styles.absoluteBox1}>
            <Text style={styles.boxText}>Absolute 1</Text>
          </View>
          <View style={styles.absoluteBox2}>
            <Text style={styles.boxText}>Absolute 2</Text>
          </View>
          <Text style={styles.relativeText}>
            Relative content that flows normally
          </Text>
        </View>
      </View>

      {/* Z-index layering */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Z-Index Layering</Text>
        <View style={styles.layeredContainer}>
          <View style={[styles.layer, styles.layer1]}>
            <Text style={styles.boxText}>Layer 1 (z: 1)</Text>
          </View>
          <View style={[styles.layer, styles.layer2]}>
            <Text style={styles.boxText}>Layer 2 (z: 3)</Text>
          </View>
          <View style={[styles.layer, styles.layer3]}>
            <Text style={styles.boxText}>Layer 3 (z: 2)</Text>
          </View>
        </View>
      </View>

      {/* Negative margins */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Negative Margins</Text>
        <View style={styles.negativeMarginContainer}>
          <View style={styles.normalBox}>
            <Text style={styles.boxText}>Normal Box</Text>
          </View>
          <View style={styles.negativeMarginBox}>
            <Text style={styles.boxText}>Negative Margin</Text>
          </View>
          <View style={styles.normalBox}>
            <Text style={styles.boxText}>Normal Box</Text>
          </View>
        </View>
      </View>

      {/* Aspect ratio */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Aspect Ratio</Text>
        <View style={styles.aspectRatioContainer}>
          <View style={styles.aspectRatioBox}>
            <Text style={styles.boxText}>16:9 Aspect Ratio</Text>
          </View>
        </View>
      </View>

      {/* Complex alignment */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Complex Alignment</Text>
        <View style={styles.complexAlignmentContainer}>
          <View style={styles.cornerBox1}>
            <Text style={styles.smallText}>Top Left</Text>
          </View>
          <View style={styles.cornerBox2}>
            <Text style={styles.smallText}>Top Right</Text>
          </View>
          <View style={styles.centerBox}>
            <Text style={styles.boxText}>Center</Text>
          </View>
          <View style={styles.cornerBox3}>
            <Text style={styles.smallText}>Bottom Left</Text>
          </View>
          <View style={styles.cornerBox4}>
            <Text style={styles.smallText}>Bottom Right</Text>
          </View>
        </View>
      </View>

      {/* Overflow handling */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Overflow Handling</Text>
        <View style={styles.overflowContainer}>
          <Text style={styles.overflowText}>
            This is a very long text that will be clipped because the container
            has overflow hidden and a fixed height. The text continues beyond
            the visible area.
          </Text>
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f8f9fa',
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
  },
  section: {
    marginBottom: 30,
    backgroundColor: 'white',
    borderRadius: 8,
    padding: 16,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 12,
    color: '#333',
  },
  relativeContainer: {
    height: 120,
    backgroundColor: '#e9ecef',
    borderRadius: 4,
    padding: 16,
    position: 'relative',
  },
  absoluteBox1: {
    position: 'absolute',
    top: 10,
    right: 10,
    backgroundColor: '#dc3545',
    padding: 8,
    borderRadius: 4,
  },
  absoluteBox2: {
    position: 'absolute',
    bottom: 10,
    left: 10,
    backgroundColor: '#28a745',
    padding: 8,
    borderRadius: 4,
  },
  relativeText: {
    color: '#333',
    textAlign: 'center',
    marginTop: 20,
  },
  layeredContainer: {
    height: 150,
    position: 'relative',
  },
  layer: {
    position: 'absolute',
    width: 100,
    height: 60,
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 4,
  },
  layer1: {
    backgroundColor: '#007bff',
    top: 20,
    left: 20,
    zIndex: 1,
  },
  layer2: {
    backgroundColor: '#dc3545',
    top: 40,
    left: 60,
    zIndex: 3,
  },
  layer3: {
    backgroundColor: '#28a745',
    top: 60,
    left: 40,
    zIndex: 2,
  },
  negativeMarginContainer: {
    backgroundColor: '#e9ecef',
    padding: 16,
    borderRadius: 4,
  },
  normalBox: {
    backgroundColor: '#6c757d',
    padding: 16,
    marginVertical: 8,
    borderRadius: 4,
    alignItems: 'center',
  },
  negativeMarginBox: {
    backgroundColor: '#ffc107',
    padding: 16,
    marginVertical: -8, // Negative margin
    marginHorizontal: 20,
    borderRadius: 4,
    alignItems: 'center',
  },
  aspectRatioContainer: {
    width: '100%',
  },
  aspectRatioBox: {
    width: '100%',
    aspectRatio: 16 / 9,
    backgroundColor: '#6f42c1',
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 4,
  },
  complexAlignmentContainer: {
    height: 200,
    backgroundColor: '#e9ecef',
    borderRadius: 4,
    position: 'relative',
  },
  cornerBox1: {
    position: 'absolute',
    top: 8,
    left: 8,
    backgroundColor: '#dc3545',
    padding: 8,
    borderRadius: 4,
  },
  cornerBox2: {
    position: 'absolute',
    top: 8,
    right: 8,
    backgroundColor: '#28a745',
    padding: 8,
    borderRadius: 4,
  },
  cornerBox3: {
    position: 'absolute',
    bottom: 8,
    left: 8,
    backgroundColor: '#ffc107',
    padding: 8,
    borderRadius: 4,
  },
  cornerBox4: {
    position: 'absolute',
    bottom: 8,
    right: 8,
    backgroundColor: '#17a2b8',
    padding: 8,
    borderRadius: 4,
  },
  centerBox: {
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: [{ translateX: -50 }, { translateY: -25 }],
    backgroundColor: '#6f42c1',
    padding: 16,
    borderRadius: 4,
  },
  overflowContainer: {
    height: 60,
    backgroundColor: '#e9ecef',
    borderRadius: 4,
    padding: 16,
    overflow: 'hidden',
  },
  overflowText: {
    fontSize: 16,
    color: '#333',
  },
  boxText: {
    color: 'white',
    fontWeight: 'bold',
    textAlign: 'center',
  },
  smallText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
});
```

## Custom Components

### Reusable Button Component
```javascript
import React from 'react';
import {
  TouchableOpacity,
  Text,
  StyleSheet,
  ActivityIndicator,
  View,
} from 'react-native';

function CustomButton({
  title,
  onPress,
  variant = 'primary',
  size = 'medium',
  disabled = false,
  loading = false,
  icon,
  iconPosition = 'left',
  fullWidth = false,
  style,
  textStyle,
  ...props
}) {
  const variants = {
    primary: {
      backgroundColor: '#007bff',
      borderColor: '#007bff',
      textColor: '#ffffff',
    },
    secondary: {
      backgroundColor: 'transparent',
      borderColor: '#6c757d',
      textColor: '#6c757d',
    },
    success: {
      backgroundColor: '#28a745',
      borderColor: '#28a745',
      textColor: '#ffffff',
    },
    danger: {
      backgroundColor: '#dc3545',
      borderColor: '#dc3545',
      textColor: '#ffffff',
    },
    warning: {
      backgroundColor: '#ffc107',
      borderColor: '#ffc107',
      textColor: '#212529',
    },
    outline: {
      backgroundColor: 'transparent',
      borderColor: '#007bff',
      textColor: '#007bff',
    },
  };

  const sizes = {
    small: {
      paddingVertical: 8,
      paddingHorizontal: 12,
      fontSize: 14,
    },
    medium: {
      paddingVertical: 12,
      paddingHorizontal: 16,
      fontSize: 16,
    },
    large: {
      paddingVertical: 16,
      paddingHorizontal: 20,
      fontSize: 18,
    },
  };

  const variantStyle = variants[variant];
  const sizeStyle = sizes[size];

  const buttonStyle = [
    styles.button,
    {
      backgroundColor: variantStyle.backgroundColor,
      borderColor: variantStyle.borderColor,
      paddingVertical: sizeStyle.paddingVertical,
      paddingHorizontal: sizeStyle.paddingHorizontal,
    },
    fullWidth && styles.fullWidth,
    disabled && styles.disabled,
    style,
  ];

  const buttonTextStyle = [
    styles.buttonText,
    {
      color: variantStyle.textColor,
      fontSize: sizeStyle.fontSize,
    },
    disabled && styles.disabledText,
    textStyle,
  ];

  const renderContent = () => {
    if (loading) {
      return (
        <View style={styles.loadingContainer}>
          <ActivityIndicator
            size="small"
            color={variantStyle.textColor}
            style={styles.loadingIndicator}
          />
          <Text style={buttonTextStyle}>Loading...</Text>
        </View>
      );
    }

    if (icon) {
      return (
        <View style={[
          styles.contentContainer,
          iconPosition === 'right' && styles.contentContainerReverse
        ]}>
          {React.cloneElement(icon, {
            color: variantStyle.textColor,
            size: sizeStyle.fontSize,
          })}
          <Text style={[buttonTextStyle, styles.iconText]}>{title}</Text>
        </View>
      );
    }

    return <Text style={buttonTextStyle}>{title}</Text>;
  };

  return (
    <TouchableOpacity
      style={buttonStyle}
      onPress={onPress}
      disabled={disabled || loading}
      activeOpacity={0.7}
      {...props}
    >
      {renderContent()}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  button: {
    borderRadius: 6,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: 44,
  },
  buttonText: {
    fontWeight: '600',
    textAlign: 'center',
  },
  fullWidth: {
    width: '100%',
  },
  disabled: {
    opacity: 0.6,
  },
  disabledText: {
    color: '#6c757d',
  },
  loadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  loadingIndicator: {
    marginRight: 8,
  },
  contentContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  contentContainerReverse: {
    flexDirection: 'row-reverse',
  },
  iconText: {
    marginLeft: 8,
    marginRight: 8,
  },
});

export default CustomButton;
```

### Advanced Input Component
```javascript
import React, { useState, forwardRef, useImperativeHandle, useRef } from 'react';
import {
  View,
  TextInput,
  Text,
  TouchableOpacity,
  Animated,
  StyleSheet,
} from 'react-native';

const CustomInput = forwardRef(({
  label,
  placeholder,
  value,
  onChangeText,
  onFocus,
  onBlur,
  error,
  helperText,
  leftIcon,
  rightIcon,
  secureTextEntry,
  multiline = false,
  numberOfLines = 1,
  maxLength,
  editable = true,
  required = false,
  style,
  inputStyle,
  ...props
}, ref) => {
  const [isFocused, setIsFocused] = useState(false);
  const [isSecure, setIsSecure] = useState(secureTextEntry);
  const animatedValue = useRef(new Animated.Value(value ? 1 : 0)).current;
  const inputRef = useRef(null);

  useImperativeHandle(ref, () => ({
    focus: () => inputRef.current?.focus(),
    blur: () => inputRef.current?.blur(),
    clear: () => {
      onChangeText?.('');
      inputRef.current?.clear();
    },
    isFocused: () => inputRef.current?.isFocused(),
  }));

  const handleFocus = (e) => {
    setIsFocused(true);
    animateLabel(1);
    onFocus?.(e);
  };

  const handleBlur = (e) => {
    setIsFocused(false);
    if (!value) {
      animateLabel(0);
    }
    onBlur?.(e);
  };

  const animateLabel = (toValue) => {
    Animated.timing(animatedValue, {
      toValue,
      duration: 200,
      useNativeDriver: false,
    }).start();
  };

  const labelStyle = {
    position: 'absolute',
    left: leftIcon ? 40 : 12,
    top: animatedValue.interpolate({
      inputRange: [0, 1],
      outputRange: [multiline ? 20 : 16, 4],
    }),
    fontSize: animatedValue.interpolate({
      inputRange: [0, 1],
      outputRange: [16, 12],
    }),
    color: animatedValue.interpolate({
      inputRange: [0, 1],
      outputRange: ['#6c757d', isFocused ? '#007bff' : '#6c757d'],
    }),
  };

  const toggleSecureEntry = () => {
    setIsSecure(!isSecure);
  };

  const containerStyle = [
    styles.container,
    isFocused && styles.containerFocused,
    error && styles.containerError,
    !editable && styles.containerDisabled,
    style,
  ];

  const textInputStyle = [
    styles.textInput,
    leftIcon && styles.textInputWithLeftIcon,
    rightIcon && styles.textInputWithRightIcon,
    multiline && styles.textInputMultiline,
    inputStyle,
  ];

  return (
    <View style={styles.wrapper}>
      <View style={containerStyle}>
        {leftIcon && (
          <View style={styles.leftIconContainer}>
            {leftIcon}
          </View>
        )}

        {label && (
          <Animated.Text style={labelStyle}>
            {label}{required && ' *'}
          </Animated.Text>
        )}

        <TextInput
          ref={inputRef}
          style={textInputStyle}
          placeholder={!label ? placeholder : undefined}
          placeholderTextColor="#6c757d"
          value={value}
          onChangeText={onChangeText}
          onFocus={handleFocus}
          onBlur={handleBlur}
          secureTextEntry={isSecure}
          multiline={multiline}
          numberOfLines={multiline ? numberOfLines : 1}
          maxLength={maxLength}
          editable={editable}
          {...props}
        />

        {rightIcon && (
          <TouchableOpacity style={styles.rightIconContainer}>
            {rightIcon}
          </TouchableOpacity>
        )}

        {secureTextEntry && (
          <TouchableOpacity
            style={styles.rightIconContainer}
            onPress={toggleSecureEntry}
          >
            <Text style={styles.eyeIcon}>
              {isSecure ? '👁️' : '🙈'}
            </Text>
          </TouchableOpacity>
        )}
      </View>

      {(error || helperText) && (
        <View style={styles.helperContainer}>
          <Text style={[
            styles.helperText,
            error && styles.errorText
          ]}>
            {error || helperText}
          </Text>
          {maxLength && value && (
            <Text style={styles.counterText}>
              {value.length}/{maxLength}
            </Text>
          )}
        </View>
      )}
    </View>
  );
});

const styles = StyleSheet.create({
  wrapper: {
    marginBottom: 16,
  },
  container: {
    borderWidth: 1,
    borderColor: '#dee2e6',
    borderRadius: 8,
    backgroundColor: '#ffffff',
    minHeight: 56,
    position: 'relative',
  },
  containerFocused: {
    borderColor: '#007bff',
    borderWidth: 2,
  },
  containerError: {
    borderColor: '#dc3545',
  },
  containerDisabled: {
    backgroundColor: '#f8f9fa',
    opacity: 0.6,
  },
  textInput: {
    fontSize: 16,
    color: '#333333',
    paddingHorizontal: 12,
    paddingTop: 20,
    paddingBottom: 8,
    flex: 1,
  },
  textInputWithLeftIcon: {
    paddingLeft: 40,
  },
  textInputWithRightIcon: {
    paddingRight: 40,
  },
  textInputMultiline: {
    textAlignVertical: 'top',
    minHeight: 80,
  },
  leftIconContainer: {
    position: 'absolute',
    left: 12,
    top: 16,
    zIndex: 1,
  },
  rightIconContainer: {
    position: 'absolute',
    right: 12,
    top: 16,
    zIndex: 1,
  },
  eyeIcon: {
    fontSize: 18,
  },
  helperContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 4,
    paddingHorizontal: 4,
  },
  helperText: {
    fontSize: 12,
    color: '#6c757d',
    flex: 1,
  },
  errorText: {
    color: '#dc3545',
  },
  counterText: {
    fontSize: 12,
    color: '#6c757d',
  },
});

export default CustomInput;
```

## Layout Systems

### Grid System
```javascript
import React from 'react';
import { View, Text, StyleSheet, Dimensions } from 'react-native';

const { width } = Dimensions.get('window');

function GridSystem() {
  const GridRow = ({ children, spacing = 8 }) => (
    <View style={[styles.row, { marginHorizontal: -spacing / 2 }]}>
      {React.Children.map(children, child =>
        React.cloneElement(child, { spacing })
      )}
    </View>
  );

  const GridCol = ({ children, size = 1, spacing = 8 }) => {
    const colWidth = (width - 40 - (spacing * 11)) / 12; // 40 for container padding
    const columnWidth = colWidth * size + spacing * (size - 1);
    
    return (
      <View style={[
        styles.col,
        {
          width: columnWidth,
          marginHorizontal: spacing / 2,
        }
      ]}>
        {children}
      </View>
    );
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Grid System</Text>

      <GridRow>
        <GridCol size={12}>
          <View style={styles.gridItem}>
            <Text style={styles.gridText}>12 columns</Text>
          </View>
        </GridCol>
      </GridRow>

      <GridRow>
        <GridCol size={6}>
          <View style={styles.gridItem}>
            <Text style={styles.gridText}>6 columns</Text>
          </View>
        </GridCol>
        <GridCol size={6}>
          <View style={styles.gridItem}>
            <Text style={styles.gridText}>6 columns</Text>
          </View>
        </GridCol>
      </GridRow>

      <GridRow>
        <GridCol size={4}>
          <View style={styles.gridItem}>
            <Text style={styles.gridText}>4</Text>
          </View>
        </GridCol>
        <GridCol size={4}>
          <View style={styles.gridItem}>
            <Text style={styles.gridText}>4</Text>
          </View>
        </GridCol>
        <GridCol size={4}>
          <View style={styles.gridItem}>
            <Text style={styles.gridText}>4</Text>
          </View>
        </GridCol>
      </GridRow>

      <GridRow>
        <GridCol size={3}>
          <View style={styles.gridItem}>
            <Text style={styles.gridText}>3</Text>
          </View>
        </GridCol>
        <GridCol size={3}>
          <View style={styles.gridItem}>
            <Text style={styles.gridText}>3</Text>
          </View>
        </GridCol>
        <GridCol size={3}>
          <View style={styles.gridItem}>
            <Text style={styles.gridText}>3</Text>
          </View>
        </GridCol>
        <GridCol size={3}>
          <View style={styles.gridItem}>
            <Text style={styles.gridText}>3</Text>
          </View>
        </GridCol>
      </GridRow>

      <GridRow>
        <GridCol size={2}>
          <View style={styles.gridItem}>
            <Text style={styles.gridText}>2</Text>
          </View>
        </GridCol>
        <GridCol size={8}>
          <View style={styles.gridItem}>
            <Text style={styles.gridText}>8</Text>
          </View>
        </GridCol>
        <GridCol size={2}>
          <View style={styles.gridItem}>
            <Text style={styles.gridText}>2</Text>
          </View>
        </GridCol>
      </GridRow>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#f8f9fa',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
  },
  row: {
    flexDirection: 'row',
    marginBottom: 8,
  },
  col: {
    // Column styles handled dynamically
  },
  gridItem: {
    backgroundColor: '#007bff',
    padding: 16,
    borderRadius: 4,
    alignItems: 'center',
    minHeight: 50,
    justifyContent: 'center',
  },
  gridText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 12,
  },
});
```

## Animations

### Basic Animations
```javascript
import React, { useRef, useEffect, useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  Animated,
  StyleSheet,
  Easing,
} from 'react-native';

function BasicAnimations() {
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(-100)).current;
  const scaleAnim = useRef(new Animated.Value(1)).current;
  const rotateAnim = useRef(new Animated.Value(0)).current;
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    // Auto-start animations
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 1000,
        useNativeDriver: true,
      }),
      Animated.spring(slideAnim, {
        toValue: 0,
        tension: 100,
        friction: 8,
        useNativeDriver: true,
      }),
    ]).start();
  }, []);

  const handleFadeToggle = () => {
    Animated.timing(fadeAnim, {
      toValue: isVisible ? 0 : 1,
      duration: 500,
      useNativeDriver: true,
    }).start();
    setIsVisible(!isVisible);
  };

  const handlePulse = () => {
    Animated.sequence([
      Animated.timing(scaleAnim, {
        toValue: 1.2,
        duration: 150,
        useNativeDriver: true,
      }),
      Animated.timing(scaleAnim, {
        toValue: 1,
        duration: 150,
        useNativeDriver: true,
      }),
    ]).start();
  };

  const handleRotate = () => {
    Animated.timing(rotateAnim, {
      toValue: 1,
      duration: 1000,
      easing: Easing.linear,
      useNativeDriver: true,
    }).start(() => {
      rotateAnim.setValue(0);
    });
  };

  const handleShake = () => {
    const shakeAnimation = Animated.sequence([
      Animated.timing(slideAnim, { toValue: 10, duration: 50, useNativeDriver: true }),
      Animated.timing(slideAnim, { toValue: -10, duration: 50, useNativeDriver: true }),
      Animated.timing(slideAnim, { toValue: 10, duration: 50, useNativeDriver: true }),
      Animated.timing(slideAnim, { toValue: -10, duration: 50, useNativeDriver: true }),
      Animated.timing(slideAnim, { toValue: 0, duration: 50, useNativeDriver: true }),
    ]);
    shakeAnimation.start();
  };

  const spin = rotateAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '360deg'],
  });

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Basic Animations</Text>

      {/* Fade Animation */}
      <Animated.View style={[
        styles.animatedBox,
        styles.fadeBox,
        { opacity: fadeAnim }
      ]}>
        <Text style={styles.boxText}>Fade Animation</Text>
      </Animated.View>

      {/* Slide Animation */}
      <Animated.View style={[
        styles.animatedBox,
        styles.slideBox,
        { transform: [{ translateX: slideAnim }] }
      ]}>
        <Text style={styles.boxText}>Slide Animation</Text>
      </Animated.View>

      {/* Scale Animation */}
      <Animated.View style={[
        styles.animatedBox,
        styles.scaleBox,
        { transform: [{ scale: scaleAnim }] }
      ]}>
        <Text style={styles.boxText}>Scale Animation</Text>
      </Animated.View>

      {/* Rotate Animation */}
      <Animated.View style={[
        styles.animatedBox,
        styles.rotateBox,
        { transform: [{ rotate: spin }] }
      ]}>
        <Text style={styles.boxText}>Rotate</Text>
      </Animated.View>

      {/* Control Buttons */}
      <View style={styles.buttonContainer}>
        <TouchableOpacity style={styles.button} onPress={handleFadeToggle}>
          <Text style={styles.buttonText}>Toggle Fade</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.button} onPress={handlePulse}>
          <Text style={styles.buttonText}>Pulse Scale</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.button} onPress={handleRotate}>
          <Text style={styles.buttonText}>Rotate</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.button} onPress={handleShake}>
          <Text style={styles.buttonText}>Shake</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#f8f9fa',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 30,
  },
  animatedBox: {
    width: 150,
    height: 60,
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
    alignSelf: 'center',
  },
  fadeBox: {
    backgroundColor: '#007bff',
  },
  slideBox: {
    backgroundColor: '#28a745',
  },
  scaleBox: {
    backgroundColor: '#dc3545',
  },
  rotateBox: {
    backgroundColor: '#ffc107',
  },
  boxText: {
    color: 'white',
    fontWeight: 'bold',
    textAlign: 'center',
  },
  buttonContainer: {
    marginTop: 30,
  },
  button: {
    backgroundColor: '#6c757d',
    padding: 12,
    borderRadius: 8,
    marginBottom: 8,
    alignItems: 'center',
  },
  buttonText: {
    color: 'white',
    fontWeight: 'bold',
  },
});
```

## Gestures

### Basic Gesture Handling
```javascript
import React, { useRef, useState } from 'react';
import {
  View,
  Text,
  PanGestureHandler,
  TapGestureHandler,
  PinchGestureHandler,
  State,
} from 'react-native-gesture-handler';
import Animated, {
  useAnimatedGestureHandler,
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  runOnJS,
} from 'react-native-reanimated';

function GestureExample() {
  const translateX = useSharedValue(0);
  const translateY = useSharedValue(0);
  const scale = useSharedValue(1);
  const [gestureInfo, setGestureInfo] = useState('');

  const updateGestureInfo = (info) => {
    setGestureInfo(info);
  };

  // Pan Gesture Handler
  const panGestureHandler = useAnimatedGestureHandler({
    onStart: (_, context) => {
      context.startX = translateX.value;
      context.startY = translateY.value;
      runOnJS(updateGestureInfo)('Pan started');
    },
    onActive: (event, context) => {
      translateX.value = context.startX + event.translationX;
      translateY.value = context.startY + event.translationY;
      runOnJS(updateGestureInfo)(`Panning: ${Math.round(event.translationX)}, ${Math.round(event.translationY)}`);
    },
    onEnd: () => {
      translateX.value = withSpring(0);
      translateY.value = withSpring(0);
      runOnJS(updateGestureInfo)('Pan ended - returning to center');
    },
  });

  // Pinch Gesture Handler
  const pinchGestureHandler = useAnimatedGestureHandler({
    onStart: () => {
      runOnJS(updateGestureInfo)('Pinch started');
    },
    onActive: (event) => {
      scale.value = event.scale;
      runOnJS(updateGestureInfo)(`Pinching: ${event.scale.toFixed(2)}x`);
    },
    onEnd: () => {
      scale.value = withSpring(1);
      runOnJS(updateGestureInfo)('Pinch ended - returning to normal size');
    },
  });

  // Double Tap Handler
  const doubleTapRef = useRef();
  const singleTapRef = useRef();

  const handleSingleTap = () => {
    setGestureInfo('Single tap detected');
  };

  const handleDoubleTap = () => {
    scale.value = withSpring(scale.value === 1 ? 1.5 : 1);
    setGestureInfo('Double tap detected - toggling scale');
  };

  const animatedStyle = useAnimatedStyle(() => {
    return {
      transform: [
        { translateX: translateX.value },
        { translateY: translateY.value },
        { scale: scale.value },
      ],
    };
  });

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Gesture Handling</Text>
      
      <Text style={styles.info}>{gestureInfo || 'Try gestures on the box below'}</Text>

      <View style={styles.gestureArea}>
        <TapGestureHandler
          ref={singleTapRef}
          onActivated={handleSingleTap}
          waitFor={doubleTapRef}
        >
          <TapGestureHandler
            ref={doubleTapRef}
            numberOfTaps={2}
            onActivated={handleDoubleTap}
          >
            <PinchGestureHandler onGestureEvent={pinchGestureHandler}>
              <PanGestureHandler onGestureEvent={panGestureHandler}>
                <Animated.View style={[styles.gestureBox, animatedStyle]}>
                  <Text style={styles.gestureText}>
                    Drag, Pinch, Tap, or Double Tap me!
                  </Text>
                </Animated.View>
              </PanGestureHandler>
            </PinchGestureHandler>
          </TapGestureHandler>
        </TapGestureHandler>
      </View>

      <View style={styles.instructions}>
        <Text style={styles.instructionTitle}>Instructions:</Text>
        <Text style={styles.instruction}>• Drag to move the box</Text>
        <Text style={styles.instruction}>• Pinch to scale the box</Text>
        <Text style={styles.instruction}>• Single tap for feedback</Text>
        <Text style={styles.instruction}>• Double tap to toggle size</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#f8f9fa',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
  },
  info: {
    fontSize: 16,
    textAlign: 'center',
    marginBottom: 20,
    color: '#666',
    minHeight: 20,
  },
  gestureArea: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#e9ecef',
    borderRadius: 12,
    margin: 20,
  },
  gestureBox: {
    width: 150,
    height: 150,
    backgroundColor: '#007bff',
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 4,
    },
    shadowOpacity: 0.3,
    shadowRadius: 6,
    elevation: 8,
  },
  gestureText: {
    color: 'white',
    fontWeight: 'bold',
    textAlign: 'center',
    fontSize: 16,
  },
  instructions: {
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 8,
    marginTop: 20,
  },
  instructionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  instruction: {
    fontSize: 14,
    marginBottom: 4,
    color: '#666',
  },
});

export default GestureExample;
```

---

*Continue to: [12-react-native-navigation.md](./12-react-native-navigation.md)*
