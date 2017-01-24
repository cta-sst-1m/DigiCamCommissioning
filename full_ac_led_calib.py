
## First run analyse_hvoff script

mycommand = [os.environ['GORCAROOT'] + '/Run/submit.py', 'care', fullName,

             '-c' ,os.environ['GORCAROOT' ] + '/care_cfg/' +careconfigfile,
'-r', str(numberofrun),
'-l', os.environ['GORCAROOT' ] + '/Run/' +batchconfig,
'-o' ,outdir,
'-e' , "NSBRATEPERPIXEL 0 " +str(nsb),
'-e' , "BIASCURVETRIALS " +str(evtpertrial),
'-e' , 'BIASCURVESTART ' +str(scanrange[0]),
'-e' , 'BIASCURVESTOP ' +str(scanrange[1]),
'-e' , 'BIASCURVESTEP ' +str(scanrange[2])]
if triggertype == 'snapshot': mycommand+= ['-e', 'USETRIGGERCAMERASNAPSHOT 0 1',
'-e', 'SNAPSHOTPATCHCIRCLES 0 '+str ( Mpattern[M]),
'-e', 'SNAPSHOTSCALINGDIVISOR 0 '+str ( D),
'-e', 'SNAPSHOTSCOMBINATION 0 0 '+str ( N)]
elif triggertype == 'nextneighbor':
mycommand+= [ '-e', 'USETRIGGERCAMERASNAPSHOT 0 0',
'-e', 'GROUPMULTIPLICITY 0 '+str ( M)]

for e in extraoptions:
    mycommand+=[ '-e',e] print
mycommand
mystring = ''
for s in mycommand:
    mystring += s+' ' print 'Running'
,mystring
p = Popen(mycommand, stdout=PIPE, stdin=PIPE, stderr=STDOUT, bufsize=1)
jobids = []
for line in iter(p.stdout.readline, ''):
    print line, \
          p.stdout \
.close()



# TODO : save the chi2 of the fits, stode in database
