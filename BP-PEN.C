//Neuron structure for solving XOR PROBLEM USING
// BP Training Algorithm

# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <conio.h>
# include <time.h>
# include <dos.h>

# define numInp 17
# define numHid 10
# define numOut 5
# define numPat 125
FILE *f1;
float mu=0.0005,a=-0.5,b=0.5,n1;
void generatewt(int);
float rmserr();
void test();
//struct time t1,t2;
int x[numPat][numInp],y[numPat][numOut];
float weight[numHid*numInp+numHid*numOut],err=50.0;
float wo[numHid][numOut],wh[numInp][numHid];
float dwo[numHid][numOut],dwh[numInp][numHid];
float net,fnet1[numHid],fpnet1[numHid],fnet2[numOut];
float e1[numOut],e2[numOut],fpnet2[numOut],eh1[numHid],eh2[numHid];
int i,k,j,pat,p;
int t;
long int ep;
void main()
{
  clrscr();
//  gettime(&t1);

//  test();

/*      randomize();
   do
    {
       n=rand();
       i++;
    } while(i<100);
   i=0;j=0;
*/

  //INITIATE WEIGHTS
  generatewt(numHid*(numInp+1)+numHid*numOut);
  k=0;
  for(i=0;i<=numInp;i++)
  {
     for(j=0;j<numHid;j++)
     {
     wh[i][j]=weight[k++];
     }
  }
  for(i=0;i<numHid;i++)
  {
     for(j=0;j<numOut;j++)
     {
       wo[i][j]=weight[k++];
     }
  }

  f1=fopen("train.dat","r");
  for(p=0;p<125;p++)
  {
    for(i=0;i<16;i++)
    {
      fscanf(f1,"%d",&t);
      x[p][i]=t;
 //     printf("%d ",t);
    }
     x[p][16]=1.0;
    for(i=0;i<5;i++)
    {
      fscanf(f1,"%d",&t);
      y[p][i]=t;
    }
  }
  fclose(f1);

  ep=0;
  while(err>0.05)
  {
    ep++;
    for(i=0;i<numHid;i++)
    {
       for(j=0;j<numOut;j++)
       {
	 dwo[i][j]=0.0;
       }
    }
    for(i=0;i<numInp;i++)
    {
       for(j=0;j<numHid;j++)
       {
	 dwh[i][j]=0.0;
       }
    }
    for(pat=0;pat<numPat;pat++)
    {
      for(i=0;i<numHid;i++)
      {
	 net=0.0;
	 for(j=0;j<numInp;j++)
	   net += x[pat][j] * wh[j][i];
	 fnet1[i]=1.0/(1.0+exp(-net));
	 fpnet1[i]=fnet1[i]*(1-fnet1[i]);
      }
      for(i=0;i<numOut;i++)
      {
	 net=0.0;
	 for(j=0;j<numHid;j++)
	   net+=fnet1[j]*wo[j][i];
	 fnet2[i]=1.0/(1.0+exp(-net));
	 fpnet2[i]=fnet2[i]*(1-fnet2[i]);
	 e1[i]=y[pat][i]-fnet2[i];
      }
      for(i=0;i<numHid;i++)
      {
	eh1[i]=0.0;
	for(j=0;j<numOut;j++)
	{
	  dwo[i][j]=mu*fpnet2[j]*e1[j]*fnet1[i];
	  eh1[i]=eh1[i]+(wo[i][j]*e1[j]*fpnet2[j]);
	}
      }
      for(i=0;i<numHid;i++)
      {
	  for(j=0;j<numInp;j++)
	  {
	    dwh[j][i]=mu*x[pat][j]*fpnet1[i]*eh1[i];
	  }
     }
     for(i=0;i<numHid;i++)
     {
       for(j=0;j<numOut;j++)
       {
	  wo[i][j]=wo[i][j]+dwo[i][j];
       }
     }

     for(i=0;i<numHid;i++)
     {
       for(j=0;j<numInp;j++)
       {
	 wh[j][i]=wh[j][i]+dwh[j][i];
       }
     }
   }
   err=rmserr();
     printf("\n %f  %ld ",err,ep);
  }
//     gettime(&t2);
  printf("\n MSE= %f Epoch= %ld \n",err,ep);
  test();
//   printf("\nEnter Time : %2d:%02d:%02d:%02d\n",t1.ti_hour,t1.ti_min,t1.ti_sec,t1.ti_hund);
//   printf("Exit Time : %2d:%02d:%02d:%02d\n  ",t2.ti_hour,t2.ti_min,t2.ti_sec,t2.ti_hund);


  getch();
}
void generatewt(int num)
{
   int i=0;
   float n;
   do
   {
//      n=rand();
      n=rand()/(float)RAND_MAX;
      n=(b-a)*n+a;
      if(n!=0.0)
      {
	weight[i]=n;
	i++;
      }
   }while(i<num);
}
float rmserr()
{
   int c;
    float ac,err=0.0,mse=0.0,f1[numHid],f2[numOut];
    for(c=0;c<numPat;c++)
     {
       for(i=0;i<numHid;i++)
       {
	 net=0.0;
	 for(j=0;j<numInp;j++)
	   net += x[c][j] * wh[j][i];
	 f1[i]=1.0/(1.0+exp(-net));
       }
       for(i=0;i<numOut;i++)
       {
	 net=0.0;
	 for(j=0;j<numHid;j++)
	   net+=f1[j]*wo[j][i];
	 f2[i]=1.0/(1.0+exp(-net));
	 err=y[c][i]-f2[i];
	 mse=mse+err*err;
       }
     }
     mse=mse/(2*numPat);
     return(mse);
 }
void test()
{
   FILE *f2,*f3;
   int c,t;
   float ac,f1[numHid],y2;
   int xt[125][17],yt[125][5];
   f2=fopen("test.dat","r");
  for(p=0;p<125;p++)
  {
    for(i=0;i<16;i++)
    {
      fscanf(f2,"%d",&t);
      xt[p][i]=t;
 //     printf("%d ",t);
    }
     xt[p][16]=1.0;
    for(i=0;i<5;i++)
    {
      fscanf(f2,"%d",&t);
      yt[p][i]=t;
    }
  }
  fclose(f2);
    f3=fopen("penresult.dat","w");
    for(c=0;c<125;c++)
     {
       for(i=0;i<numHid;i++)
       {
	 ac=0.0;
	 for(j=0;j<numInp;j++)
	   ac += xt[c][j] * wh[j][i];
	 f1[i]=1.0/(1.0+exp(-ac));
       }
      for(i=0;i<numOut;i++)
      {
	 ac=0.0;
	 for(j=0;j<numHid;j++)
	   ac+=f1[j]*wo[j][i];
	 y2=1.0/(1.0+exp(-ac));
	 fprintf(f3," E = %d Ob = %f for  %d \n",yt[c][i],y2,c);
      }
      fprintf(f3,"\n");
   }
   fclose(f3);
   return;
}

