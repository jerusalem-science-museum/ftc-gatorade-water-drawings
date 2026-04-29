#include "Routines.h"

enum serialState { GET_PARAMS,
                   GET_CMD,
                   GET_CASSETTE,
                   GET_BYTES };
serialState st;
int bytepos;
int num_bytes_to_read;  // set at GET_PARAMS.
byte data;
char ser_msg[100];
void setup() {
  st = GET_PARAMS;
  bytepos = 0;
  Serial.begin(BAUDRATE);
  delay(1000);  // wait to make sure serial begin
  pinMode(SR_en_pin, OUTPUT);
  digitalWrite(SR_en_pin, HIGH);
  pinMode(SR_st_pin, OUTPUT);
  pinMode(SR_clk_pin, OUTPUT);
  pinMode(SR_data_pin, OUTPUT);
  digitalWrite(SR_st_pin, LOW);
  digitalWrite(SR_clk_pin, LOW);
  digitalWrite(SR_data_pin, LOW);
  off_all_valves(num_of_valves);
  pulse_io(SR_st_pin);
  pinMode(red_led_pin, OUTPUT);
  pinMode(green_led_pin, OUTPUT);
  pinMode(blue_led_pin, OUTPUT);
  led_off();

  digitalWrite(SR_en_pin, LOW);
  got_param = false;
  Serial.println("START");
}

void get_params() {
  byte value;
  value = Serial.read();
  Serial.println(value);
  switch (param_index) {
    case 0: image_h = value; break;
    case 1: valve_on_time = value; break;
    case 2: drawing_depth = value; break;
    default: Serial.println("ERROR: GOT TOO MUCH PARAMETERS"); break;
  }
  param_index++;
  if (param_index >= PARAM_NUMBER) {
    got_param = true;
    param_index = 0;
    Serial.println("got all parameters - Im good to go");
    num_bytes_to_read = image_h * image_w / 8;
    st = GET_CMD;
  }
}

void loop() {

  // handle serial data.
  if (Serial.available() > 0) {
    switch (st) {
      case GET_PARAMS:
        get_params();
        break;
      case GET_CMD:
        if (drawing_flag)  // still drawing.
          break;
        data = Serial.read();
        if (data == DROP_KEY)
          init_drawing();
        else if (data == START_KEY) 
        {
          Serial.println("got start");
          st = GET_CASSETTE;
        } 
        else 
        {
          Serial.print("got unknown cmd: ");
          Serial.println(data);
        }
        break;
      case GET_CASSETTE:
        cassette_drawing = Serial.read();
        Serial.println("got cassette value");
        Serial.println(cassette_drawing);
        sprintf(ser_msg, "expecting %d bytes (%d X %d / 8)", num_bytes_to_read, image_h, image_w);
        Serial.println(ser_msg);
        st = GET_BYTES;
        break;
      case GET_BYTES:
        data = Serial.read();
        if (data == END_KEY) {
          if (bytepos < num_bytes_to_read || bytepos > num_bytes_to_read) {
            sprintf(ser_msg, "got %d bytes instead of %d (%d x %d / 8)", bytepos, num_bytes_to_read, image_h, image_w);
            Serial.println(ser_msg);
          } else {
            sprintf(ser_msg, "got %d bytes as expected (+ end key).", num_bytes_to_read);
            Serial.println(ser_msg);
            init_drawing();
            st = GET_CMD;
          }
          bytepos = 0;
        } else {
          image[bytepos++] = data;
          Serial.println(data);
          if ((bytepos) % 8 == 0)
            Serial.println(GOOD_KEY);
        }
        break;
    }
  }

  if (valve_on_flag && millis() - last_valve_on > valve_on_time) {
    off_all_valves(num_of_valves);  // without pulsing ST because layers should be continuous
    valve_on_flag = false;
  }
  if (led_on_flag && millis() - last_led_on > led_on_time) {
    if (!full_light)
      led_off();
    led_on_flag = false;
  }
  if (led_start_flag && millis() - last_led_start > led_start) {
    led_start_flag = false;
    led_on_flag = true;
    last_led_on = millis();
    if (!full_light && led_on_time > 0)
      led_on(color);
  }

  if (drawing_flag) {
    if (check_drawing()) {
      drawing_flag = false;
      off_all_valves(num_of_valves);
      pulse_io(SR_st_pin);
      color += 1;
      if (color >= colors_num)
        color = 0;
      Serial.println(READY_KEY);
    }
  }
}
