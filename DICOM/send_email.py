def send_email(Subject, Text, From='nam_research@naver.com', To='sunnmoon137@gmail.com'):
    import smtplib
    from email.mime.text import MIMEText

    pwd = '~!Q@W#E$R'
    msg = MIMEText(Text, _charset='euc-kr')
    msg['Subject'] = Subject
    msg['From'] = From
    msg['To'] = To

    try:
        server = smtplib.SMTP('smtp.naver.com', 587)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(From, pwd)
        server.sendmail(From, To, msg.as_string())
        server.quit()

    except smtplib.SMTPException as e:
        print e