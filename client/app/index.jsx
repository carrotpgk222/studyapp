import { useRouter } from "expo-router";
import { Text, View, StyleSheet, Image, TouchableOpacity, Dimensions } from "react-native";

const { width } = Dimensions.get("window");

export default function HomeScreen() {
  const router = useRouter(); // Use expo-router's navigation

  return (
    <View style={styles.container}>
      <Image source={{ uri: "https://i.imgur.com/FLXaWSm.png" }} style={styles.logo} />

      <Text style={styles.title}>
        <Text style={styles.highlight}>Unlock</Text> Your <Text style={styles.highlight}>Potential</Text>
      </Text>

      <Text style={styles.description}>
        Manage your time, improve your education, and get responsive feedback.{"\n"}
        Customize your learning journey today!
      </Text>

      <Text style={styles.slogan}>Deeply Seeking Knowledge, One Step at a Time</Text>

      <View style={styles.buttonContainer}>
        <TouchableOpacity
          style={styles.loginButton}
          onPress={() => router.push("/login")} // Navigate to login screen
        >
          <Text style={styles.buttonText}>Log in</Text>
        </TouchableOpacity>
        <TouchableOpacity 
          style={styles.signupButton}
          onPress={() => router.push("/register")} // Correct navigation for expo-router
        >
          <Text style={styles.buttonText}>Sign up</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f3e7e9",
    paddingHorizontal: 20,
  },
  logo: {
    width: 450,
    height: 240,
    alignSelf: "center",
    padding: 60,
  },
  title: {
    fontSize: 32,
    fontWeight: "bold",
    textAlign: "center",
    marginTop: 50,
  },
  highlight: {
    color: "#318ba4",
  },
  description: {
    fontSize: 18,
    color: "#222",
    textAlign: "center",
    marginTop: 40,
    paddingHorizontal: 20,
  },
  slogan: {
    fontSize: 16,
    color: "#444",
    fontStyle: "italic",
    textAlign: "center",
    marginTop: 30,
  },
  buttonContainer: {
    flexDirection: "row",
    position: "absolute",
    bottom: 0,
    width: width,
    height: 80,
    justifyContent: "center",
    alignItems: "stretch",
  },
  loginButton: {
    flex: 1,
    backgroundColor: "rgba(255, 255, 255, 0.9)",
    justifyContent: "center",
    alignItems: "center",
    height: 60,
  },
  signupButton: {
    flex: 1,
    backgroundColor: "#318ba4",
    justifyContent: "center",
    alignItems: "center",
    height: 60,
  },
  buttonText: {
    fontSize: 18,
    fontWeight: "bold",
  },
});