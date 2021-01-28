
import numpy as np
from flask import Flask, request, render_template, Response
import models as dbHandler
import mailgenerator
import re
import os
import pickle
from joblib import load
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from skimage.transform import resize
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from werkzeug.utils import secure_filename
from flask_mysqldb import MySQL
import MySQLdb.cursors
from camera import VideoCamera
from picture import PhotoCamera

app = Flask(__name__)
userName=''
otp=None
mysql = MySQL(app)

def put_user(name):
    global userName
    userName=name
    return True

def get_user():
    global userName
    return userName

ruser=''
rpass=''
remail=''
rnum=''
lemail=''

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')


@app.route('/userlogin')
def userloginpage():
    return render_template('userlogin.html')


@app.route('/register', methods =['GET', 'POST'])
def register():
    if request.method=='POST':
        global username
        username=request.form['username']
        password=request.form['password']
        email=request.form['email']
        phone=request.form['phone']
        username=username.strip()
        email=email.strip()
        n=put_user(username)
        echeck=dbHandler.regemail(email)
        ucheck=dbHandler.reguser(username)
        if echeck==True and ucheck==True:
            x=[]
            global ruser
            global rpass
            global remail
            global rnum
            ruser = username
            rpass = password
            remail = email
            rnum = phone
            x.append(email)
            global otp
            otp=mailgenerator.Sending_report(x)
            return render_template('mailverification.html', email=email)
        elif ucheck!=True:
            return render_template('register.html',info=ucheck)
        elif echeck!=True:
            return render_template('register.html',info=echeck)
    else:
        return render_template('register.html')


@app.route('/mailverification', methods=['GET','POST'])
def mailverification():
    if request.method=='POST':
        global otp
        motp=request.form['vemail']
        motp=int(motp)
        if otp==motp:
            global ruser
            global rpass
            global remail
            global rnum
            dbHandler.insertUser(ruser,rpass,remail,rnum)
            return render_template('register.html' ,info='Registeration Successful!')
        else:
            return render_template('mailverification.html' ,email=remail, info='Wrong OTP')


@app.route('/passmail')
def mailpage():
    return render_template('passmail.html')


@app.route('/passmail', methods=['GET','POST'])
def passmail():
    if request.method=='POST':
        email=request.form['email']
        x=[]
        global otp
        global remail
        x.append(email)
        otp=mailgenerator.Sending_report(x)
        remail = email
        return render_template('passreset.html' ,email=email)



@app.route('/passreset')
def resetpage():
    return render_template('passreset.html')

@app.route('/changepass')
def changepage():
    return render_template('changepass.html')

@app.route('/passreset', methods=['GET','POST'])
def passreset():
    if request.method=='POST':
        motp=request.form['vemail']
        password=request.form['password']
        cpass=request.form['cpass']
        global otp
        global remail
        motp=int(motp)
        if otp==motp and password==cpass:
            dbHandler.resetpass(remail,password)
            return render_template('userlogin.html' ,info='Password Reset Successful!')
        elif otp!=motp:
            return render_template('passreset.html' ,info='Wrong OTP')
        elif password!=cpass:
            return render_template('passreset.html' ,info='Check your password again')
    else:
        return render_template('passreset.html')


@app.route('/changepass', methods=['GET','POST'])
def changepass():
    if request.method=='POST':
        email=request.form['email']
        password=request.form['password']
        cpass=request.form['cpass']
        global lemail
        if lemail==email and password==cpass:
            dbHandler.changepass(email,password)
            return render_template('changepass.html', info="Password updated successfully!")
        elif lemail!=email:
            return render_template('changepass.html', info="Wrong mail id")
        elif password!=cpass:
            return render_template('changepass.html', info="Check your password again")




@app.route('/addusers', methods =['GET', 'POST'])
def addinguser():
    if request.method=='POST':
        global username
        username=request.form['username']
        password=request.form['password']
        email=request.form['email']
        phone=request.form['phone']
        valid=dbHandler.checkuser(username,password,email,phone)
        if valid==True:
            return render_template('addusers.html',info='')
        else:
            x=[]
            dbHandler.insertUser(username,password,email,phone)
            x.append(email)
            mailgenerator.Sending_report(x)
            return render_template('addusers.html',info='Registration Successful!')
    else:
        return render_template('addusers.html')


@app.route('/deleteuser', methods =['GET','POST'])
def deleteuser():
    if request.method=='POST':
        global username
        username=request.form['username']
        email=request.form['email']
        x=dbHandler.deletinguser(username,email)
        if x==True:
            return render_template('deletinguser.html',info='Deletion Successful!')
        else:
            return render_template('deletinguser.html',info=x)


@app.route('/userlogin',methods=['POST','GET'])
def userlogin():
    if request.method=='POST':
        global username
        username=request.form['username']
        n=put_user(username)
        password=request.form['password']
        valid=dbHandler.check(username,password)
        if valid==True:
            global lemail
            lemail=dbHandler.getmail(username,password)
            return render_template('selection.html',user=username)
        else:
            return render_template('userlogin.html',info=valid)
    else:
        return render_template('userlogin.html')


@app.route('/adminview')
def adminview():
    global username
    x=dbHandler.retrieveUsers()
    return render_template('adminview.html',user=username,userslist=x)

@app.route('/adminlogin')
def adminloginpage():
    return render_template('adminlogin.html')

@app.route('/deletinguser')
def deletinguser():
    return render_template('deletinguser.html')

@app.route('/adminlogin',methods=['POST','GET'])
def adminlogin():
    if request.method=='POST':
        global username
        username=request.form['username']
        password=request.form['password']
        valid=dbHandler.checkadmin(username,password)
        if valid==True:
            x=dbHandler.retrieveUsers()
            return render_template('adminview.html',user=username,userslist=x)
        else:
            return render_template('adminlogin.html',info=valid)
    else:
        render_template('adminlogin.html',info=valid)


@app.route('/select')
def select():
    username=get_user()
    return render_template('selection.html',user=username)


@app.route('/picture')
def picture():
    username=get_user()
    return render_template('picture.html',user=username)



def pgenerate(picture):
    global file_path
    frame = picture.get_pframe(file_path)
    yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def pcartoon(picture):
    car = picture.get_pcartoon(file_path)
    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + car + b'\r\n\r\n')


# def pemoji(picture):
#     emj = picture.get_pemoji()
#     yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + emj + b'\r\n\r\n')




@app.route('/upload', methods=['GET', 'POST'])
def upload():
    # print('*******************')
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # print('++++++++++++++++++++++')
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        global file_path
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        print(file_path)
        f.save(file_path)
        # print('--------------------------')
        # return render_template('display.html')
        username=get_user()
        return render_template('display.html',user=username)



@app.route('/result', methods=['GET', 'POST'])
def result():
    username=get_user()
    return render_template('display.html',user=username)


@app.route('/photo_feed')
def photo_feed():
    return Response(pgenerate(PhotoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/photo_cartoon')
def photo_cartoon():
    return Response(pcartoon(PhotoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/photo_emj')
# def photo_emj():
#     return Response(pemoji(PhotoCamera()),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')
#




@app.route('/addemployee')
def add():
    return render_template('addemployee.html')


@app.route('/selection')
def selection():
    username=get_user()
    return render_template('selection.html',user=username)


@app.route('/index')
def index():
    username=get_user()
    return render_template('index.html',user=username)


def generate(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def cartoon(camera):
     while True:
        car = camera.get_cartoon()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + car + b'\r\n\r\n')


def emoji(camera):
     while True:
        emj = camera.get_emoji()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + emj + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_cartoon')
def video_cartoon():
    return Response(cartoon(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_emj')
def video_emj():
    return Response(emoji(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == '__main__':
    app.run(debug=True)
