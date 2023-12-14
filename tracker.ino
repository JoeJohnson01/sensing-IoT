#include <ArduinoBLE.h>
#include <Arduino_LSM9DS1.h>

// Define color constants
#define COLOR_RED    1
#define COLOR_GREEN  2
#define COLOR_BLUE   3

// Define LED pins
#define LED_RED    22 
#define LED_GREEN  23
#define LED_BLUE   24


#define MAX_DATA_POINTS 3000

// Service name to be broadcasted
#define PERIPHERAL_NAME "Lift Sensor Right"
#define SERVICE_UUID "LSR-Service-UUID"
#define CHARACTERISTIC_INPUT_UUID "LSR-Input-UUID"
#define CHARACTERISTIC_OUTPUT_UUID "LSR-Output-UUID"

// Output characteristic used to send the response back to the connected phone
BLECharacteristic outputChar(CHARACTERISTIC_OUTPUT_UUID, BLENotify, 512);

void setLEDColor(int color, bool on) {
    digitalWrite(LED_RED, color == COLOR_RED && on ? LOW : HIGH);
    digitalWrite(LED_GREEN, color == COLOR_GREEN && on ? LOW : HIGH);
    digitalWrite(LED_BLUE, color == COLOR_BLUE && on ? LOW : HIGH);
}

// Function to blink the LED
void blinkLED(int color, int interval, int duration) {
    long startTime = millis();
    while (millis() - startTime < duration) {
        setLEDColor(color, true);
        delay(interval);
        setLEDColor(color, false);
        delay(interval);
    }
}

struct IMUData {
    float ax, ay, az;
    float gx, gy, gz;
    float mx, my, mz;
    unsigned long timestamp;
};

class IMUReader {
private:
    IMUData data[MAX_DATA_POINTS];
    int dataIndex = 0;
    bool collecting = false;

public:
    void start() {
        dataIndex = 0;
        collecting = true;
    }

    void stop() {
        collecting = false;
    }

    void update() {
        if (collecting && dataIndex < MAX_DATA_POINTS) {
            if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable() && IMU.magneticFieldAvailable()) {
                IMUData currentData;
                currentData.timestamp = millis();
                IMU.readAcceleration(currentData.ax, currentData.ay, currentData.az);
                IMU.readGyroscope(currentData.gx, currentData.gy, currentData.gz);
                IMU.readMagneticField(currentData.mx, currentData.my, currentData.mz);
                data[dataIndex++] = currentData;
            }
        }
    }

    IMUData* getData() {
        return data;
    }

    int getDataSize() {
        return dataIndex;
    }
};

IMUReader imuReader;

void setup() {
    pinMode(LED_RED, OUTPUT);
    pinMode(LED_GREEN, OUTPUT);
    pinMode(LED_BLUE, OUTPUT);

    Serial.begin(115200);

    if (!IMU.begin()) {
        Serial.println("Failed to initialize IMU!");
        blinkLED(COLOR_RED, 500, 5000);
        return;
    } else {
        Serial.println("IMU initialized.");
    }

    if (!BLE.begin()) {
        Serial.println("Failed to initialize BLE!");
        blinkLED(COLOR_RED, 500, 5000);
        return;
    } else {
        Serial.println("BLE initialized.");
    }

    BLE.setLocalName(PERIPHERAL_NAME);
    BLE.setAdvertisedServiceUuid(SERVICE_UUID);

    BLEService imuService(SERVICE_UUID);
    BLECharacteristic inputChar(CHARACTERISTIC_INPUT_UUID, BLEWrite, 512);

    imuService.addCharacteristic(inputChar);
    imuService.addCharacteristic(outputChar);
    BLE.addService(imuService);

    inputChar.setEventHandler(BLEWritten, [](BLEDevice central, BLECharacteristic characteristic) {
        String inputValue = String((const char*)characteristic.value());
        if (inputValue.length() > 0) {
            Serial.println("Received: " + inputValue);
            if (inputValue == "start") {
                setLEDColor(COLOR_GREEN, true);
                imuReader.start();
            } else if (inputValue == "stop") {
                setLEDColor(COLOR_BLUE, true);
                blinkLED(COLOR_GREEN, 100, 1000);
                imuReader.stop();

                auto dataSize = imuReader.getDataSize();
                for (int i = 0; i < dataSize; i++) {
                    const auto& d = imuReader.getData()[i];
                    String dataString = String(d.timestamp) + "," + String(d.ax) + "," + String(d.ay) + "," + String(d.az) + "," + String(d.gx) + "," + String(d.gy) + "," + String(d.gz) + "," + String(d.mx) + "," + String(d.my) + "," + String(d.mz);
                    outputChar.writeValue(dataString.c_str());
                    delay(100);
                }
                setLEDColor(COLOR_BLUE, false);
            }
        }
    });

    BLE.advertise();
    Serial.println("Bluetooth device active, waiting for connections...");
}

void loop() {
    BLE.poll();
    imuReader.update();
    delay(100);
}
