import pickle, smtplib

def send_email(msg):
    with open('email.p', 'r') as f:
        tokken = pickle.load(f)

    fromaddr = tokken['from']
    toaddrs  = tokken['to']
    username = tokken['from']
    password = tokken['pass']
    msg = 'From: %s\nTo: %s\nSubject: %s\n\n%s' % (fromaddr, toaddrs, 'python', msg)
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.ehlo()
    server.starttls()
    server.login(username,password)
    server.sendmail(fromaddr, toaddrs, msg)
    server.quit()

if __name__ == '__main__':
    send_email('salam test with python')
