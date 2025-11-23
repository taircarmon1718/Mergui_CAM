# PTZ Camera Controller for B016712MP

## Hardware Conncetion

![Alt text](../data/HardwareConnection.png)

## install dependencies

* sudo apt update
* sudo apt install -y libatlas-base-dev python3-opencv python3-picamera2
* sudo apt install python3-numpy



## Download the source code 

```bash
git clone https://github.com/ArduCAM/PTZ-Camera-Controller.git
```


## Enable the camera module

* Edit the configuration file: **sudo nano /boot/config.txt**
* Find the line: **camera_auto_detect=1**, update it to: **camera_auto_detect=0**
* Add the entry under the line [all] at the bottom: **dtoverlay=imx477**

* Save and exit

## Enable i2c

<!-- * cd PTZ-Camera-Controller
* sudo chmod +x enable_i2c_vc.sh
* ./enable_i2c_vc.sh
Press Y to reboot -->
1. `sudo raspi-config`

2. Select **Interface Options** and enter

![select Interface Option](../data/select%20interface%20options.png)

3. Select I2C and enter

![select i2c](../data/select%20i2c.png)

4. Select YES and press **enter** to confirm

![enable i2c](../data/enable%20i2c.png)

5. exit and reboot your Pi to take effect


## Run the FocuserExample.py

* cd PTZ-Camera-Controller/B016712MP
* python3 FocuserExample.py


> Please note that after opening the program, press the `T` key first and wait for about 8 seconds. The mode will switch from '**Fix**' to '**Adjust**'. At this point, you can use the keyboard to control Zoom, Focus, IR-CUT, etc.


![Alt text](../data/Arducam%20Controller1.png)

## Run the AutofocusTableExample.py

* cd PTZ-Camera-Controller/B016712MP
* python3 AutofocusTableExample.py

> If you're running the program for the first time, place the camera in the desired position and wait five minutes for the camera to generate a focus chart.


![Alt text](../data/Focuser%20AutoFocus.png)

### Generate autofocus configuration

The program will automatically read the autofocus file when it starts. If the file is not available, it will prompt the user to generate the autofocus configuration using the dedicated program.

When entering the program to generate the auto-zoom focus configuration, please ensure that the camera is fixed on the desired area for photography.

If the resulting configuration does not yield satisfactory results, press F to regenerate the configuration.


> Note:
>
> - Compared to FocuserExample.py, the AutofocusTableExample.py provides smoother autofocus. By simply pressing the `↑` and `↓` arrow keys, you can automatically adjust the focus and zoom.
>
> - If your keyboard keys are not functioning, please make sure that you are not in the "Fix" mode. If so, Press `T` to switch to Adjust mode.
> - If you are not satisfied with the autofocus performance, you can press the `R` key to reset the autofocus table and then press the 'F' key to regenerate the autofocus table.



<!-- ## Datasheet

<style type="text/css">
.tg  {border-collapse:collapse;border-color:#ccc;border-spacing:0;}
.tg td{background-color:#fff;border-color:#ccc;border-style:solid;border-width:1px;color:#333;
  font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{background-color:#f0f0f0;border-color:#ccc;border-style:solid;border-width:1px;color:#333;
  font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
.tg .tg-uzvj{border-color:inherit;font-weight:bold;text-align:center;vertical-align:middle}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-uzvj">Address (HEX)</th>
    <th class="tg-uzvj">Register Name</th>
    <th class="tg-uzvj">Description</th>
    <th class="tg-uzvj">Default Val</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-uzvj">00</td>
    <td class="tg-9wq8">Focus</td>
    <td class="tg-9wq8">Bit[15:0]:  Focus value</td>
    <td class="tg-9wq8">0x0000</td>
  </tr>
  <tr>
    <td class="tg-uzvj">01</td>
    <td class="tg-9wq8">Zoom</td>
    <td class="tg-9wq8">Bit[15:0]:  Zoom value</td>
    <td class="tg-9wq8">0x0000</td>
  </tr>
  <tr>
    <td class="tg-uzvj" rowspan="3">04</td>
    <td class="tg-9wq8" rowspan="3">Bus status</td>
    <td class="tg-9wq8">Bit[15:1]: Reserved</td>
    <td class="tg-9wq8" rowspan="3">0x0000</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[0]: 1: BUSY</td>
  </tr>
  <tr>
    <td class="tg-9wq8">         0: IDLE</td>
  </tr>
  <tr>
    <td class="tg-uzvj" rowspan="2">05</td>
    <td class="tg-9wq8" rowspan="2">Pan</td>
    <td class="tg-9wq8">Range:  0~180</td>
    <td class="tg-9wq8" rowspan="2">0x005a</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[15:0]:  Pan value</td>
  </tr>
  <tr>
    <td class="tg-uzvj" rowspan="2">06</td>
    <td class="tg-9wq8" rowspan="2">Tilt</td>
    <td class="tg-9wq8">Range:  0~180</td>
    <td class="tg-9wq8" rowspan="2">0x005a</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[15:0]:  Tilt value</td>
  </tr>
  <tr>
    <td class="tg-uzvj">07</td>
    <td class="tg-9wq8">Focus maximum</td>
    <td class="tg-9wq8">Bit[15:0]:  Focus Max value</td>
    <td class="tg-9wq8">0x0834</td>
  </tr>
  <tr>
    <td class="tg-uzvj">08</td>
    <td class="tg-9wq8">Zoom maximum</td>
    <td class="tg-9wq8">Bit[15:0]:  Zoom Max value</td>
    <td class="tg-9wq8">0x0834</td>
  </tr>
  <tr>
    <td class="tg-uzvj" rowspan="2">0A</td>
    <td class="tg-9wq8" rowspan="2">Reset focus</td>
    <td class="tg-9wq8">Bit[15:1]: Reserved</td>
    <td class="tg-9wq8" rowspan="2">0x0000</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[0]: 1: Reset active</td>
  </tr>
  <tr>
    <td class="tg-uzvj" rowspan="2">0B</td>
    <td class="tg-9wq8" rowspan="2">Reset zoom</td>
    <td class="tg-9wq8">Bit[15:1]: Reserved</td>
    <td class="tg-9wq8" rowspan="2">0x0000</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[0]: 1: Reset active</td>
  </tr>
  <tr>
    <td class="tg-uzvj" rowspan="4">0C</td>
    <td class="tg-9wq8" rowspan="4">IR  cut control</td>
    <td class="tg-9wq8">Bit[15:1]: Reserved</td>
    <td class="tg-9wq8" rowspan="4">0x0000</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[0]: 1: ON</td>
  </tr>
  <tr>
    <td class="tg-9wq8">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0: OFF</td>
  </tr>
  <tr>
    <td class="tg-9wq8">based on real IR cut device</td>
  </tr>
  <tr>
    <td class="tg-uzvj" rowspan="3">0E</td>
    <td class="tg-9wq8" rowspan="3">Pan&amp;Tilt</td>
    <td class="tg-9wq8">Range:  0~180</td>
    <td class="tg-9wq8" rowspan="3">0x5a5a</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[15:8]: Pan value</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[0:7]: Tilt value</td>
  </tr>
  <tr>
    <td class="tg-uzvj" rowspan="2">0F</td>
    <td class="tg-9wq8" rowspan="2">Focus &amp; Zoom</td>
    <td class="tg-9wq8">Bit[31:16]:  Focus value</td>
    <td class="tg-9wq8" rowspan="2">0x00000000</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[15:0 ]:   Zoom value</td>
  </tr>
  <tr>
    <td class="tg-uzvj" rowspan="2">11</td>
    <td class="tg-9wq8" rowspan="2">Reset focus&amp;zoom</td>
    <td class="tg-9wq8">Bit[15:1]: Reserved</td>
    <td class="tg-9wq8" rowspan="2">0x0000</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[0]: 1: Reset active</td>
  </tr>
  <tr>
    <td class="tg-uzvj" rowspan="3">30</td>
    <td class="tg-9wq8" rowspan="3">Operation mode</td>
    <td class="tg-9wq8">Bit[15:1]:Reserved</td>
    <td class="tg-9wq8" rowspan="3">0x0001</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[0]: 1: Adjust mode</td>
  </tr>
  <tr>
    <td class="tg-9wq8">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0: Fixed mode</td>
  </tr>
  <tr>
    <td class="tg-uzvj" rowspan="22">50~65</td>
    <td class="tg-9wq8" rowspan="22">Focus &amp; Zoom table</td>
    <td class="tg-9wq8">Bit[15:0 ]: Zoom max step</td>
    <td class="tg-9wq8">0xff</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[15:0 ]: Focus max step</td>
    <td class="tg-9wq8">0xff</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[15:0 ]: 1x zoom val</td>
    <td class="tg-9wq8">0xff</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[15:0 ]: 1x focus val</td>
    <td class="tg-9wq8">0xff</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[15:0 ]: 2x zoom val</td>
    <td class="tg-9wq8">0xff</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[15:0 ]: 2x focus val</td>
    <td class="tg-9wq8">0xff</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[15:0 ]: 3x zoom val</td>
    <td class="tg-9wq8">0xff</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[15:0 ]: 3x focus val</td>
    <td class="tg-9wq8">0xff</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[15:0 ]: 4x zoom val</td>
    <td class="tg-9wq8">0xff</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[15:0 ]: 4x focus val</td>
    <td class="tg-9wq8">0xff</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[15:0 ]: 5x zoom val</td>
    <td class="tg-9wq8">0xff</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[15:0 ]: 5x focus val</td>
    <td class="tg-9wq8">0xff</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[15:0 ]: 6x zoom val</td>
    <td class="tg-9wq8">0xff</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[15:0 ]: 6x focus val</td>
    <td class="tg-9wq8">0xff</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[15:0 ]: 7x zoom val</td>
    <td class="tg-9wq8">0xff</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[15:0 ]: 7x focus val</td>
    <td class="tg-9wq8">0xff</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[15:0 ]: 8x zoom val</td>
    <td class="tg-9wq8">0xff</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[15:0 ]: 8x focus val</td>
    <td class="tg-9wq8">0xff</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[15:0 ]: 9x zoom val</td>
    <td class="tg-9wq8">0xff</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[15:0 ]: 9x focus val</td>
    <td class="tg-9wq8">0xff</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[15:0 ]: 10x zoom val</td>
    <td class="tg-9wq8">0xff</td>
  </tr>
  <tr>
    <td class="tg-9wq8">Bit[15:0 ]: 10x focus val</td>
    <td class="tg-9wq8">0xff</td>
  </tr>
</tbody>
</table> -->






## Refering the link to get more information about the PTZ-Camera-Controller

[Pan/Tilt/Zoom Camera](http://www.arducam.com/docs/cameras-for-raspberry-pi/ptz-camera/)


