from django.shortcuts import render
import random
# Create your views here.
SENT_LENGTH = 150
WROD_LENGTH = 70
def index(request):
 # Render the HTML template index.html with the data in the context variable
   return render(request, 'index.html')

def tokenize(s):
   word_list = s.split()[:WROD_LENGTH]
   return word_list
   
def report(request):
   # Render the HTML template index.html with the data in the context variable
   # year = request.POST.get('year')
   report = request.FILES['finfile']
   name  = report.name
   content = report.read()
   sent_list =  content.decode("utf-8").split('\n')[:SENT_LENGTH]
   sent_list = list(map(tokenize, sent_list))
   
   dict_list = []
   
   for sent in sent_list:
      value_list = [random.random() for _ in range(len(sent))]
      w_dict = dict(zip(sent,value_list))
      dict_list.append(w_dict)

   sent_value_list = ([random.random() for _ in range(SENT_LENGTH)])
   report = dict(zip(sent_value_list,dict_list))
   index_report = dict(zip([i for i in range(1, len(dict_list)+1)],dict_list))
   return render(request, 'report2.html', {'name':name,'report':report, 'index_report':index_report})