import sqlite3 as sql

def insertUser(username,password,email,phone):
    con = sql.connect("database.sqlite")
    cur = con.cursor()
    cur.execute("INSERT INTO Logins (username,password) VALUES (?,?)", (username,password))
    cur.execute("INSERT INTO Register (username,password,email,mobile) VALUES (?,?,?,?)", (username,password,email,phone))
    con.commit()
    con.close()

def retrieveUsers():
    userslist={}
    con=sql.connect("database.sqlite")
    cur=con.cursor()
    cur.execute("SELECT username,email,mobile,password FROM Register")
    users=cur.fetchall()
    cnt=1
    for i,j,k,l in users:
        userslist[cnt]=(i,j,k,l)
        cnt+=1
    con.close()
    return userslist

def deletinguser(username,email):
    con = sql.connect("database.sqlite")
    try:
        cur = con.cursor()
        valid=checkusers(username,email)
        if valid==True:
            cur.execute("delete from Logins where username=?",(username,))
            cur.execute("delete from Register where username=? AND email=?",(username,email))
            con.commit()
            con.close()
            return True
        else:
            return valid
    except:
        return valid


def check(name,pwd):
    con = sql.connect("database.sqlite")
    cur = con.cursor()
    cur.execute("SELECT username FROM Logins WHERE username=?",(name,))
    user=cur.fetchone()
    if user is not None:
        cur.execute("Select username From Logins WHERE (username=? AND password=?)",(name,pwd))
        res=cur.fetchone()
        con.close()
        if res is not None:
            return True
        else:
            return 'Invalid password'
    else:
        return 'Invalid user'


def checkuser(name,pwd,email,phone):
    con = sql.connect("database.sqlite")
    cur = con.cursor()
    cur.execute("SELECT username FROM Register WHERE (username=? AND email=?)",(name,email))
    user=cur.fetchone()
    if user is not None:
        cur.execute("SELECT username FROM Register WHERE (username=? AND email=?)",(name,email))
        res=cur.fetchone()
        con.close()
        if res is not None:
            return True
        else:
            return 'Invalid password'
    else:
        return 'Invalid user'


def regemail(remail):
    con = sql.connect("database.sqlite")
    cur = con.cursor()
    cur.execute("SELECT email FROM Register WHERE email=?",(remail,))
    remail=cur.fetchone()
    con.close()
    if remail is not None:
        return 'Email already exists'
    else:
        return True


def reguser(user):
    con = sql.connect("database.sqlite")
    cur = con.cursor()
    cur.execute("SELECT username FROM Register WHERE username=?",(user,))
    ruser=cur.fetchone()
    con.close()
    if ruser is not None:
        return 'Username already exists'
    else:
        return True


def getmail(user,pwd):
    con = sql.connect("database.sqlite")
    cur = con.cursor()
    cur.execute("SELECT email FROM Register WHERE username=? AND password=?",(user,pwd))
    lemail=cur.fetchone()
    con.close()
    lemail=lemail[0]
    return lemail



def checkusers(name,email):
    con = sql.connect("database.sqlite")
    cur = con.cursor()
    cur.execute("SELECT username FROM Register WHERE username=? ",(name,))
    user=cur.fetchone()
    # print(user,'user')
    if user is not None:
        cur.execute("SELECT username From Register WHERE (username=? AND email=?)",(name,email))
        res=cur.fetchone()
        con.close()
        if res is not None:
            return True
        else:
            return 'Invalid email'
    else:
        return 'Invalid user'

def checkadmin(username,password):
    con = sql.connect("database.sqlite")
    cur = con.cursor()
    cur.execute("SELECT username FROM Admin WHERE username=?",(username,))
    ad=cur.fetchone()
    if ad is not None:
        cur.execute("SELECT username FROM Admin WHERE (username=? AND password=?)",(username,password))
        res=cur.fetchone()
        con.close()
        if res is not None:
            return True
        else:
            return 'Invalid Password'
    else:
        return 'Invalid User'



def resetpass(remail,pwd):
    con = sql.connect("database.sqlite")
    cur = con.cursor()
    cur.execute("SELECT username FROM Register WHERE email=?",(remail,))
    user=cur.fetchone()
    user=user[0]
    cur.execute("UPDATE Logins SET password = ? WHERE  username=?",(pwd,user))
    cur.execute("UPDATE Register SET password = ? WHERE  username=? AND email=?",(pwd,user,remail))
    con.commit()
    con.close()

def changepass(lemail,pwd):
    con = sql.connect("database.sqlite")
    cur = con.cursor()
    cur.execute("SELECT username FROM Register WHERE email=?",(lemail,))
    user=cur.fetchone()
    user=user[0]
    cur.execute("UPDATE Logins SET password = ? WHERE  username=?",(pwd,user))
    cur.execute("UPDATE Register SET password = ? WHERE  username=? AND email=?",(pwd,user,lemail))
    con.commit()
    con.close()
