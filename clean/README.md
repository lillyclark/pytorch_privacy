This folder contains the code for Privacy-Utility Trade Studies in Mobile Network Data Obfuscation Scheme Design.

The script privacy.py contains the setup of all four privatizers. In the __main__ function, the developer can select which privatizer to test, and running this script will vary the parameter specific to that privatizer and record data to a csv file according to the privatizer name.

The data for these experiments is contained in the zipped file augmented\_data.csv.tar.gz. It contains GSM measurements from 9 users in the city of Chania over approximately 8 months. This dataset comes from the thesis work of  Emmanouil Alimpertis. Here is a short description for each of the dataset variables:

1. timestamp: It contains the exact date and the time moment of the measurement.

2. dt1: rrsi1 was recorded at the time moment timestamp+dt1 . Each measurement contains a burst of 5 consecutive RSS measurements from the iPhone's baseband. System records 5 consecutive RSSIs for many reasons. For example, we can see if the RSS is constant during a few seconds; this will reveal many properties of the wireless channel at the corresponding time moment.

3.dt2: timestamp+dt1+dt2 was the moment when the rssi2 was recorded.

4. and so ...

5. ...

6. ...

7. rssi\_ios: RSSI value provided by the iOS private API. rssi\_iOS is presented on the iPhone's screen if you press *3001#12345#*. However, iOS software averages rssi1 ... rssi5 during a time interval which is determined internally by the iPhone. Individually, rssi1 ... rssi5 are more accurate since they are provided by the baseband itself through AT commands and they correspond to instantaneous measurements approximately per second (usually dt\_i ~=1 sec).

8. rssi1: RSSI, in dBm, read from the iPhone's baseband at timestamp+dt1.

9. rssi2: RSSI, in dBm, read from the iPhone's baseband at timestamp+dt1+dt2.

10. rssi3: RSSI, in dBm, read from the iPhone's baseband at timestamp+dt1+dt3+dt3.

11. rssi4: RSSI, in dBm, read from the iPhone's baseband at timestamp+dt1+dt2+dt3+dt4.

12. rssi5: RSSI, in dBm, read from the iPhone's baseband at timestamp+dt1+dt2+dt3+dt4+dt5.

13. measurementLat: latitude provided by the iPhone's aGPS at the moment timestamp.

14. measurementLon: longitude provided by the iPhone's aGPS at the moment timestamp.

15. finLat: latitude provided by the iPhone's aGPS at the moment timestamp+dt1+dt2+dt3+dt4+dt5

16. finLon: longitude provided by the iPhone's aGPS at the moment timestamp+dt1+dt2+dt3+dt4+dt5. Thus, we can determine if the user was moving during the time interval (timestamp-timestamp+timestamp+dt1+dt2+dt3+dt4+dt5) when the mobile was recording the five consecutive RSSIs.

17.  accur: horizontal accuracy provided by the iPhone's aGPS. Experimentally, we determined that the accuracy provided by the mobile phone was very pessimistic.

18. isMoving: 0 if there is no difference between 13,14 and 15,16. otherwise 1.

19. cellID: the identifier of the cell

19. lac: location area code.

20. mnc: mobile network code, i.e. a specific network provider. For instance, mnc=1 corresponds to Cosmote.

21. arfcn: Absolute radio-frequency channel number.

22. freq\_dlink: downlink frequency carrier (calculated by ARFCN).

23. freq\_uplin: uplink frequency carrier (calculated by ARFCN).
