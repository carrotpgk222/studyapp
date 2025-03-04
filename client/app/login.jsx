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

export default function LoginScreen() {
  const navigation = useNavigation(); // Get navigation instance

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const handleLogin = () => {
    console.log("Logging in with:", { email, password });
    // Add your login logic here
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
          <Text style={styles.header}>Log In</Text>

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

          {/* Log In button */}
          <TouchableOpacity style={styles.submitButton} onPress={handleLogin}>
            <Text style={styles.buttonText}>Log In</Text>
          </TouchableOpacity>

          {/* Switch to Register Screen */}
          <View style={styles.registerContainer}>
            <Text style={styles.registerText}>
              Don't have an account?{" "}
              <TouchableOpacity onPress={() => navigation.navigate("register")}>
                <Text style={styles.registerLink}>Sign Up</Text>
              </TouchableOpacity>
            </Text>
          </View>
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
    top: 20,
    left: 1,
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
    top: 70,
    width: 180,
    height: 100,
    justifyContent: "center",
    alignItems: "center",
  },
  logoImage: {
    width: 327,
    height: 140,
    resizeMode: "contain",
  },
  container: {
    backgroundColor: "rgba(250, 250, 250, 0.8)",
    borderRadius: 15,
    padding: 70,
    width: "100%",
    maxWidth: 500,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 6,
    elevation: 5,
    marginTop: 120,
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
    fontSize: 16,
  },
  registerContainer: {
    marginTop: 20,
    justifyContent: "center",
    alignItems: "center",
  },
  registerText: {
    fontSize: 14,
    color: "#333",
  },
  registerLink: {
    color: "#318ba4",
    fontWeight: "bold",
  },
});