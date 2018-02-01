import httplib2

conn = httplib2.HTTPConnectionWithTimeout("sugang.snu.ac.kr")
for i in range(100):
    conn.request("GET", "/sugang/ca/number.action")
    res = conn.getresponse()
    data = res.read()
    f = open('samples/'+str(i)+'.png', 'wb+')
    f.write(data)
