##Merge pileups for multiple wells
##Gracie
import pandas as pd
import sys

drop=0
for w, path in zip(range(1,len(sys.argv[1:])+1),sys.argv[1:]):
        print(w)
        path_c=path + '.cel.gz'
        path_p=path + '.plp.gz'
        cel=pd.read_csv(path_c,compression="gzip",sep="\t")
        plp=pd.read_csv(path_p,compression="gzip",sep="\t",dtype='str')
        cel=cel.rename(columns = {'#DROPLET_ID':'DROPLET_ID'})
        plp=plp.rename(columns = {'#DROPLET_ID':'DROPLET_ID'})
        print(cel)
        print(plp)

        if w==1:
                #create empty df
                plp_new=pd.DataFrame(columns=plp.columns)
                cel_new=pd.DataFrame(columns=cel.columns)
                print(cel_new)
        #print(cel.BARCODE)
        #print(len(cel.BARCODE))
        #change cell BC to reflect well
        bc=[i.split('-',1)[0] for i in cel.BARCODE]
        new_bc=[str(i+"-"+str(w)) for i in bc]

        #change droplet id and create dict
        new_id=[str(int(i)+drop) for i in cel.DROPLET_ID]
        drop_dict=dict(zip(cel.DROPLET_ID,new_id))
        #print(drop_dict)
        drop=drop+len(bc)
        #print(new_id)

        #translate plp file to new dropid
        #print(drop_dict[18226])
        new_id_plp=[drop_dict[int(i)] for i in plp.DROPLET_ID]
        print(plp)

        #update df
        new_df=pd.DataFrame({"DROPLET_ID":new_id_plp})
        plp.update(new_df)
        print(plp)
        new_df=pd.DataFrame({"DROPLET_ID":new_id,'BARCODE':new_bc})
        cel.update(new_df)
        print(cel)

        #cat cel and plp to new merged df
        plp_new=plp_new.append(plp,ignore_index=True)
        cel_new=cel_new.append(cel,ignore_index=True)

cel_new=cel_new.rename(columns = {'DROPLET_ID':'#DROPLET_ID'})
plp_new=plp_new.rename(columns = {'DROPLET_ID':'#DROPLET_ID'})
print(plp_new)
print(cel_new)
plp_new.to_csv('merged.plp.gz',index=False,sep="\t",compression='gzip')
cel_new.to_csv('merged.cel.gz',index=False,sep="\t",compression='gzip')