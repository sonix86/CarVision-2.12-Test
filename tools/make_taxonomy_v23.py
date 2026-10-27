"""User-declared regional naming aliases; never image classification evidence."""
# Nammi01/NammiBox/DongfengBox are explicitly the same vehicle in the request.
# Keeping competing Nammi/Dongfeng labels for identical photographs is invalid.
ALIASES={'Nammi':'Dongfeng','LiXiang':'Li Auto','Lixiang':'Li Auto','NETA':'Neta','NIO':'Nio','ORA':'Ora'}
def canonical_make(name):return ALIASES.get(name,name)
DISPLAY={'Dongfeng':'Dongfeng / Nammi','Li Auto':'Li Auto (LiXiang)','Zeekr':'ZEEKR','Nio':'NIO','Ora':'ORA','Neta':'NETA'}
