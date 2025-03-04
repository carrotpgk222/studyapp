import React, { useState } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Image,
  Dimensions,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
} from "react-native";
import { useNavigation } from "@react-navigation/native"; // Import navigation hook

const { width, height } = Dimensions.get("window");

export default function RegisterScreen() {
  const navigation = useNavigation(); // Get navigation instance

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleRegister = () => {
    console.log("Registered with:", { name, email, password });
  };

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === "ios" ? "padding" : "height"}
      style={styles.body}
    >
      <ScrollView contentContainerStyle={styles.scrollContainer}>
        {/* Back Button */}
        <TouchableOpacity style={styles.backButton} onPress={() => navigation.goBack()}>
          <Text style={styles.backButtonText}>←</Text>
        </TouchableOpacity>

        {/* Logo at the Top */}
        <View style={styles.logo}>
          <Image
            source={{ uri: "https://i.imgur.com/FLXaWSm.png" }}
            style={styles.logoImage}
          />
        </View>

        <View style={styles.container}>
          <Text style={styles.header}>Create Your Account</Text>

          {/* Name input */}
          <TextInput
            style={styles.input}
            placeholder="*Name"
            value={name}
            onChangeText={setName}
          />

          {/* Email input */}
          <TextInput
            style={styles.input}
            placeholder="*Email"
            value={email}
            onChangeText={setEmail}
            keyboardType="email-address"
          />

          {/* Password input */}
          <TextInput
            style={styles.input}
            placeholder="*Password"
            value={password}
            onChangeText={setPassword}
            secureTextEntry
          />

          {/* Required information notice */}
          <Text style={styles.requiredInfo}>
            <Text style={styles.asterisk}>*</Text> Required Information
          </Text>

          {/* Register button */}
          <TouchableOpacity style={styles.submitButton} onPress={handleRegister}>
            <Text style={styles.buttonText}>Register</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  body: {
    flex: 1,
    backgroundColor: "#f3e7e9",
    alignItems: "center",
    justifyContent: "center",
  },
  scrollContainer: {
    flexGrow: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  backButton: {
    position: "absolute",
    top: 50,
    left: 20,
    backgroundColor: "#ddd",
    paddingVertical: 8,
    paddingHorizontal: 15,
    borderRadius: 5,
  },
  backButtonText: {
    fontSize: 20,
    color: "#333",
  },
  logo: {
    position: "absolute",
    top: 90,
    width: 180,
    height: 180,
    justifyContent: "center",
    alignItems: "center",
  },
  logoImage: {
    width: 340,
    height: 140,
    resizeMode: "contain",
  },
  container: {
    backgroundColor: "rgba(250, 250, 250, 0.8)",
    borderRadius: 15,
    padding: 50,
    width: "100%",
    maxWidth: 500,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 6,
    elevation: 5,
    marginTop: 180,
  },
  header: {
    fontSize: 28,
    textAlign: "center",
    color: "#4d5b6c",
    marginBottom: 30,
    fontWeight: "600",
  },
  input: {
    width: "100%",
    padding: 14,
    marginBottom: 20,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#e0e0e0",
    fontSize: 16,
    color: "#333",
    backgroundColor: "#f8f8f8",
  },
  requiredInfo: {
    fontSize: 14,
    color: "#333",
    marginTop: -10,
    marginBottom: 10,
  },
  asterisk: {
    color: "red",
  },
  submitButton: {
    backgroundColor: "#3b97b9",
    padding: 16,
    borderRadius: 8,
    alignItems: "center",
    justifyContent: "center",
    marginTop: 10,
  },
  buttonText: {
    color: "white",
    fontWeight: "bold",
    fontSize: 16,
  },
});