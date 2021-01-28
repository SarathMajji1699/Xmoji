import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime

from random import randint

def Sending_report(mails):
    print('mail is sending.....wait.')
    now=datetime.now()
    t=now.strftime("%d/%m/%Y, %H:%M:%S")
    time=str(t)
    fromaddr = "sarathmajji999@gmail.com"
    otp=randint(100000, 999999)
    print('OTP sent....',otp)
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    for i in mails:
        msg['To'] = i
    msg['Subject'] = "Get Your Own Emoji through Xmoji!"
    body = "Dear,Welcome to Xmoji! We're are glad you're here.Start using Xmoji today "+time+"\nYour OTP is   "+str(otp)
    msg.attach(MIMEText(body, 'plain'))
    filename = "report.txt"
    attachment = open("report.txt", "rb")
    p = MIMEBase('application', 'octet-stream')
    p.set_payload((attachment).read())
    encoders.encode_base64(p)
    p.add_header('Content-Disposition', "attachment; filename= %s" % filename)
    msg.attach(p) #to attach a pdf
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login(fromaddr,"**********")
    text = msg.as_string()
    cnt=0
    for i in mails:
        s.sendmail(fromaddr,i, text)
        print('welcome mail sent successfully to '+i)
        cnt+=1
    print('Total Number of mails sent',cnt)
    # s.quit()
    return otp
