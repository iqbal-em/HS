# HS
INSTRUCTION MANUAL HS DETECTION AGE AND GENDER BETA APP

DETAILS : 
	Tested : Ubuntu 20.04 and Windows 10
	Storage : 400 MB free
1.	Install Python 3.8 64 bit 
apt-get install python3

2.	Clone github 

3.	Unrar source HS_PROJECT APP
# apt-get install UNRAR
# unrar x hs_project.rar


4.	Pindah direktori ke HS_PROJECT APP
# cd hs_project 

5.	Install dependencies di requirement
# Pip install -r requirement.txt

6.	Import database hs local 
a.	Install XAMPP
1.	Download installer XAMPP on website 
# https://www.apachefriends.org/download.html



 
2.	Pindah direktori ke Donwload dan install XAMPP
# chmod +x xampp-linux-x64–7.1.22–0-installer.run
# ./xampp-linux-x64–7.1.22–0-installer.run

3.	Run mysql dan Apache Web Server
 
4.	Buat database dan import database hs_database
 

 
7.	Jika menggunakan Linux edit __init__.py di direktori source hs_project/video_stream dan tambahkan code seperti ini

import pymysql
pymysql.install_as_MySQLdb()

8.	Pindah ke direktori main project nya 
# cd [your_hs_path]/
# python3 manage.py migrate
# python3 manage.py runserver

9.	Python manage.py runserver
10.	SELESAI


