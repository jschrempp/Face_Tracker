# Bridge_Testing

Solicit x and y coordinate data from the user via the Python console.  Assemble the user input into a string in
the form of "x_value,y_value" and use the bridge to transfer this string to the Arduino side.

The Arduino side receives the string and uses the Monitor to print it out.  It also flashes the onboard LED.

A future enchancement is to parse the receid string and control two servos of a pan/tilt mechanism.

## Bricks Used

**This example does not use any Bricks.** It shows direct Router Bridge communication between Python® and Arduino.

## Hardware and Software Requirements

### Hardware

- Arduino UNO Q (x1)
- USB-C® cable (for power and programming) (x1)

### Software

- Arduino App Lab


## How to Use the Example

1. Run the App

2. Use the Python console to enter x and y values when prompted.

3. Observe the Arduino LED to see that the string came across the bridge.  Use the Arduino console to view the
values from the Arduino side.

4. In a future enhancement, send pixel coodinates as x and y values and observe pan/tilt mechanism movement.

## How it Works

Once the application is running, the device performs the following operations:

- **Managing timing and LED state in Python®.**

The Python® script uses a simple loop with timing control:

```python
    from arduino.app_utils import *
    import time
    
    led_state = False
    
    while True:
     time.sleep(1)
     led_state = not led_state
     Bridge.call("set_led_state", led_state)
```

The script toggles the LED state variable every second and sends the new state to the Arduino.

- **Exposing LED control function to Python®.**

The Arduino registers its LED control function with the Router Bridge:

```cpp
    Bridge.provide("set_led_state", set_led_state);
```

- **Controlling the hardware LED.**

The Arduino sketch handles the LED hardware control:

```cpp
    void set_led_state(bool state) {
        digitalWrite(LED_BUILTIN, state ? LOW : HIGH);
    }
```

Note that the logic is inverted (LOW for on, HIGH for off), which is typical for built-in LEDs that are wired with the cathode connected to the pin.

The high-level data flow looks like this:

```
Python® Timer Loop → Router Bridge → Arduino LED Control
```

## Understanding the Code

Here is a brief explanation of the application components:

### 🔧 Backend (`main.py`)

The Python® component manages timing and LED state logic.

- **`import time`:** Provides timing functions for controlling the blink interval.  

- **`led_state = False`:** Tracks the current LED state as a boolean variable.

- **`while True:` loop :** Creates an infinite loop that runs continuously to control the LED timing. 

- **`time.sleep(1)`:** Pauses execution for 1 second between LED state changes.

- **`led_state = not led_state`:** Toggles the LED state by inverting the boolean value.
                      
- **`Bridge.call("set_led_state")`:** Sends the new LED state to the Arduino through the Router Bridge communication system. 

### 🔧 Hardware (`sketch.ino`)

The Arduino code handles LED hardware control and sets up Bridge communication.

- **`pinMode(LED_BUILTIN, OUTPUT)`:** Configures the built-in LED pin as an output for controlling the LED state.

- **`Bridge.begin()`:** Initializes the Router Bridge communication system for receiving commands from Python®.

- **`Bridge.provide(...)`:** Registers the `set_led_state` function to be callable from the Python® script.

- **`set_led_state(bool state)`:** Controls the LED hardware with inverted logic (LOW = on, HIGH = off) typical for built-in LEDs.

- **Empty `loop()`:** The main loop remains empty since all LED control is managed by the Python® script through Bridge function calls. 
