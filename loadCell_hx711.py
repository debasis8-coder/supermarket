#belt weight 


'''import statistics
import time
from hx711 import HX711
import RPi.GPIO as GPIO

hx = HX711(5, 6)
hx.set_reference_unit(0.23)
hx.reset()
hx.tare()

print("Tare done! Place object...")

try:
    while True:
        # take 10 readings quickly and average
        readings = [hx.get_weight(1) for _ in range(10)]
        avg = statistics.mean(readings)
        print(f"Avg Weight: {avg:.2f}")
        hx.power_down()
        hx.power_up()
        time.sleep(1)

except KeyboardInterrupt:
    GPIO.cleanup()
    
    
    '''
import statistics, time, redis
from hx711 import HX711
import RPi.GPIO as GPIO

#r = redis.StrictRedis(host='localhost', port=6379, db=0)

hx = HX711(5, 6)
hx.set_reference_unit(0.23)
hx.reset()
hx.tare()

print("Tare done! Place object...")

try:
    #while True:
    readings = [hx.get_weight(1) for _ in range(10)]
    avg = statistics.mean(readings)
    print(f"Avg Weight: {avg:.2f} g")

    # Push to Redis
    #r.set("loadcell:weight", avg)

    hx.power_down()
    hx.power_up()
    time.sleep(1)

except KeyboardInterrupt:
    GPIO.cleanup()

