


/*
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                               *
 *	This program is free software; you can redistribute it and/or modify         *
 *  it under the terms of the GNU General Public License as published by         *
 *  the Free Software Foundation; either version 2 of the License, or            *
 *  (at your option) any later version.                                          *
 *                                                                               *
 *  This program is distributed in the hope that it will be useful,              *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of               *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                *
 *  GNU General Public License for more details.                                 *
 *                                                                               *
 *  You should have received a copy of the GNU General Public License            *
 *  along with this program; if not, write to the Free Software                  *
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA    *
 *                                                                               *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                               *
 *  Created by Andrea Lancichinetti on 10/04/08 (email: arg.lanci@gmail.com)     *
 *	Modified on 04/02/09                                                         *
 *	Collaborators: Santo Fortunato and Filippo Radicchi                          *
 *  Location: ISI foundation, Turin, Italy                                       *
 *	Project: Benchmarking community detection programs                           *
 *                                                                               *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 */



#include <math.h>
#include <iostream>
#include <deque>
#include <set>
#include <list>
#include <vector>
#include <map>
#include <string> 
#include <fstream>
#include <ctime>
#include <iterator>

using namespace std;


#define R2_IM1 2147483563
#define R2_IM2 2147483399
#define unlikely -214741
#define R2_AM (1.0/R2_IM1)
#define R2_IMM1 (R2_IM1-1)
#define R2_IA1 40014
#define R2_IA2 40692
#define R2_IQ1 53668
#define R2_IQ2 52774
#define R2_IR1 12211
#define R2_IR2 3791
#define R2_NTAB 32
#define R2_NDIV (1+R2_IMM1/R2_NTAB)
#define R2_EPS 1.2e-7
#define R2_RNMX (1.0-R2_EPS)



long seed;

void srand4(void) {
	seed=(long)time(NULL);
}

void srand5(int rank) {
	seed=(long)(rank);
}



double ran2(long *idum) {
	int j;
	long k;
	static long idum2=123456789;
	static long iy=0;
	static long iv[R2_NTAB];
	double temp;

	if(*idum<=0 || !iy){
		if(-(*idum)<1) *idum=1*(*idum);
		else *idum=-(*idum);
		idum2=(*idum);
		for(j=R2_NTAB+7;j>=0;j--){
			k=(*idum)/R2_IQ1;
			*idum=R2_IA1*(*idum-k*R2_IQ1)-k*R2_IR1;
			if(*idum<0) *idum+=R2_IM1;
			if(j<R2_NTAB) iv[j]=*idum;
		}
		iy=iv[0];
	}
	k=(*idum)/R2_IQ1;
	*idum=R2_IA1*(*idum-k*R2_IQ1)-k*R2_IR1;
	if(*idum<0) *idum+=R2_IM1;
	k=(idum2)/R2_IQ2;
	idum2=R2_IA2*(idum2-k*R2_IQ2)-k*R2_IR2;
	if (idum2 < 0) idum2 += R2_IM2;
	j=iy/R2_NDIV;
	iy=iv[j]-idum2;
	iv[j]=*idum;
	if(iy<1) iy+=R2_IMM1;
	if((temp=R2_AM*iy)>R2_RNMX) return R2_RNMX;
	else return temp;
}



double ran4(void) {
	double r;
	
	r=ran2(&seed);
	return(r);
}





int irand(int n) {					// returns an integer number from 0 to n
	return (int(ran4()*(n+1)));
}




// this function sets "cumulative" as the cumulative function of (1/x)^tau, with range= [min, n]
int powerlaw (int n, int min, double tau, deque<double> &cumulative) {
	
	cumulative.clear();
	double a=0;			

	for (double h=min; h<n+1; h++)
		a+= pow((1./h),tau);
	
	
	double pf=0;
	for(double i=min; i<n+1; i++) {
	
		pf+=1/a*pow((1./(i)),tau);
		cumulative.push_back(pf);
	
	}
	
	return 0;	
	
}






// it builds a network given the degree sequence d;

int config_model (deque <set<int> > &en, deque<int> &d, int min) {

	
	en.clear();
	set <int> first;
	
	for (int i=0; i<d.size(); i++)
		en.push_back(first);
		

	multimap <int, int> degree_node;
	
	for(int i=0; i<d.size(); i++)
		degree_node.insert(make_pair(d[i], i));
	
	//prints(degree_node);
	
	int var=0;

	while (degree_node.size() > 0) {
		
		multimap<int, int>::iterator itlast= degree_node.end();
		itlast--;
		
		multimap <int, int>::iterator itit= itlast;
		deque <multimap<int, int>::iterator> erasenda;
		
		int inserted=0;
		
		for (int i=0; i<itlast->first; i++) {
			
			if(itit!=degree_node.begin()) {
			
				itit--;
				
				
				en[itlast->second].insert(itit->second);
				en[itit->second].insert(itlast->second);
				inserted++;
				
				erasenda.push_back(itit);				
				
			}
			
			else
				break;
		
		}
		
		
		for (int i=0; i<erasenda.size(); i++) {
			
			
			if(erasenda[i]->first>1)
				degree_node.insert(make_pair(erasenda[i]->first - 1, erasenda[i]->second));
	
			degree_node.erase(erasenda[i]);
		
		}

		
		var+= itlast->first - inserted;
		degree_node.erase(itlast);
		
	}
	
	
	

	
	for (int u=0; u<en.size(); u++) {

		
		int stopper=0;
		while (en[u].size()<min) {
			
			if (stopper++>d.size()*d.size()) {
				
				cerr<<"ERROR: it seems there should be more links than nodes in the whole network. Change parameters (increase the number of nodes, or decrease the maximum degree...) "<<endl;
				return -1;
			
			}
			
				
			int nl=irand(en.size()-1);
					
			if (nl!=u) {
						
				en[u].insert(nl);
				en[nl].insert(u);
					
			}
				
		}
	}
	
	
	
	//cout<<"diff: "<<var<<endl;
	
	
	for(int nodo=0; nodo<d.size(); nodo++) for(int krm=0; krm<en[nodo].size(); krm++) {
				
				
		int stopper2=0;
		
		while (stopper2<d.size()) {
					
			stopper2++;
			int old_node=0;
					
			int random_mate=irand(d.size()-1);
			while (random_mate==nodo)
				random_mate=irand(d.size()-1);
					
			int nodo_h=0;
			
			if (en[nodo].insert(random_mate).second) {
				
				deque <int> nodi_esterni;
				for (set<int>::iterator it_est=en[nodo].begin(); it_est!=en[nodo].end(); it_est++) if ((*it_est)!=random_mate)
					nodi_esterni.push_back(*it_est);
							
											
						
				old_node=nodi_esterni[irand(nodi_esterni.size()-1)];
						
				en[nodo].erase(old_node);
				en[random_mate].insert(nodo);
				en[old_node].erase(nodo);

											
				deque <int> not_common;
				for (set<int>::iterator it_est=en[random_mate].begin(); it_est!=en[random_mate].end(); it_est++)
					if ((old_node!=(*it_est)) && (en[old_node].find(*it_est)==en[old_node].end()))
						not_common.push_back(*it_est);
						
							
				nodo_h=not_common[irand(not_common.size()-1)];
				
				en[random_mate].erase(nodo_h);
				en[nodo_h].erase(random_mate);
				en[nodo_h].insert(old_node);
				en[old_node].insert(nodo_h);
				
				break;
				
			}
			
			
			
			
		}
	
	
	}
	
	return 0;

}



// it computes the integral of a power law
double integral (double a, double b) {

	
	if (fabs(a+1.)>1e-10)
		return (1./(a+1.)*pow(b, a+1.));
	
	
	else
		return (log(b));

}


// it returns the average degree of a power law
double average_degree(const double &dmax, const double &dmin, const double &gamma) {

	return (1./(integral(gamma, dmax)-integral(gamma, dmin)))*(integral(gamma+1, dmax)-integral(gamma+1, dmin));

}


//bisection method to find the inferior limit, in order to have the expected average degree
double solve_dmin(const double& dmax, const double &dmed, const double &gamma) {
	
	double dmin_l=1;
	double dmin_r=dmax;
	double average_k1=average_degree(dmin_r, dmin_l, gamma);
	double average_k2=dmin_r;
	
	
	if ((average_k1-dmed>0) || (average_k2-dmed<0)) {
		
		cerr<<"ERROR: the average degree is out of range:";
		
		if (average_k1-dmed>0) {
			cerr<<"\nyou should increase the average degree (bigger than "<<average_k1<<")"<<endl; 
			cerr<<"(or decrease the maximum degree...)"<<endl;
		}
		
		if (average_k2-dmed<0) {
			cerr<<"\nyou should decrease the average degree (smaller than "<<average_k2<<")"<<endl; 
			cerr<<"(or increase the maximum degree...)"<<endl;
		}
		
		return -1;	
	}
	
		
	while (fabs(average_k1-dmed)>1e-7) {
		
		double temp=average_degree(dmax, ((dmin_r+dmin_l)/2.), gamma);
		if ((temp-dmed)*(average_k2-dmed)>0) {
			
			average_k2=temp;
			dmin_r=((dmin_r+dmin_l)/2.);
		
		}
		else {
			
			average_k1=temp;
			dmin_l=((dmin_r+dmin_l)/2.);
			
		
		}
			
		

	
	}
	
	return dmin_l;
}



// it computes the correct (i.e. discrete) average of a power law
double integer_average (int n, int min, double tau) {
	
	double a=0;

	for (double h=min; h<n+1; h++)
		a+= pow((1./h),tau);
	
	
	double pf=0;
	for(double i=min; i<n+1; i++)
		pf+=1/a*pow((1./(i)),tau)*i;
	
	return pf;

}



// this function changes the community sizes merging the smallest communities
int change_community_size(deque<int> &seq) {

	
			
	
	if (seq.size()<=1)
		return 0;
	
	int min1=0;
	int min2=0;
	
	for (int i=0; i<seq.size(); i++)		
		if (seq[i]<=seq[min1])
			min1=i;
	
	if (min1==0)
		min2=1;
	
	for (int i=0; i<seq.size(); i++)		
		if (seq[i]<=seq[min2] && seq[i]>seq[min1])
			min2=i;
	

	
	seq[min1]+=seq[min2];
	
	int c=seq[0];
	seq[0]=seq[min2];
	seq[min2]=c;
	seq.pop_front();
	
	
	
	
	return 0;
}





template <typename type>
int not_norm_histogram (deque <type> &c, ostream & out, int number_of_bins, double b1, double b2) {		

	// this should be OK
	// c is the set of data, b1 is the lower bound, b2 is the upper one (if they are equal, default limits are used)
	
	
	
	double min=double(c[0]);
	double max=double(c[0]);
	
	for (int i=0; i<c.size(); i++) {
		
		if (min>double(c[i]))
			min=double(c[i]);
		
		if (max<double(c[i]))
			max=double(c[i]);
		
	}
	
	
	
	min-=1e-6;
	max+=1e-6;
	
	
	
	if (b1!=b2) {
		
		min=b1;
		max=b2;
	
	}
		
	if (max==min)
		max+=1e-3;
	
	
	
	deque <int> hist;
	deque <double> hist2;
		
	double step=min;
	double bin=(max-min)/number_of_bins;		// bin width

	while (step<=max+2*bin) {
	
		hist.push_back(0);			
		hist2.push_back(0);			
		step+=bin;
	}
	

	
		
	
	
	for (int i=0; i<c.size(); i++) {
		
		
		
		double data=double(c[i]);
		
		if (data>min && data<=max) {
			
			int index=int((data-min)/bin);		
			
				
			hist[index]++;
			hist2[index]+=double(c[i]);
		
		}
		
	}
	
	
	for (int i=0; i<hist.size()-1; i++) {
		
		
		
				
		double x=hist2[i]/hist[i];
		double y=double(hist[i])/(c.size());
		
		if (fabs(y)>1e-10)
			out<<x<<"\t"<<y<<endl;
		
	
	}
	
	
	
			
	return 0;

}


	

template <typename type>
int log_histogram (deque <type> &c, ostream & out, int number_of_bins) {		// c is the set od data, min is the lower bound, max is the upper one
	
	
	
	double min=double(c[0]);
	double max=double(c[0]);
	
	for (int i=0; i<c.size(); i++) {
		
		if (min>double(c[i]))
			min=double(c[i]);
		
		if (max<double(c[i]))
			max=double(c[i]);
		
	}
	
	
	
	
	deque <int> hist;
	deque <double> hist2;
	deque <double> bins;
	double step=log(min);
	if (max==min)
		max++;
	
	double bin=(log(max)-log(min))/number_of_bins;		// bin width
	
		

	while (step<=log(max)+2*bin) {
	
		bins.push_back(exp(step));
		hist.push_back(0);			
		hist2.push_back(0);			
		step+=bin;
	}
	
	for (int i=0; i<c.size(); i++) {
		
		
		int index=bins.size()-1;
		for (int j=0; j<bins.size()-1; j++) if(	(fabs(double(c[i])-bins[j])<1e-7) || (	double(c[i])>bins[j]	&&	double(c[i])<bins[j+1]	)	) { // this could be done in a more efficient way
			
			index=j;
			break;
		
		}
		
		
				
		hist[index]++;
		hist2[index]+=double(c[i]);
		
	}
	
	for (int i=0; i<hist.size()-1; i++) {
		
		int h1= int(bins[i]);
		int h2= int(bins[i+1]);
		int number_of_integer=h2-h1;
		
		if (fabs(h1 - bins[i])<1e-7)
			number_of_integer++;
		
		double x=hist2[i]/hist[i];
		double y=double(hist[i])/(c.size()*number_of_integer);
		
		if (fabs(y)>1e-10)
			out<<x<<"\t"<<y<<endl;
		
		
		

		
		
		
	
	
	}
	
	
	
	return 0;

}


int histogram (deque <int> &c, ostream & out) {

	
	map <int, int> hist;
	
	for (int i=0; i<c.size(); i++) {
		
		map <int, int>::iterator itf = hist.find(c[i]);
		if (itf!=hist.end())
			(*itf).second++;
		else
			hist.insert(make_pair(c[i],1));

	}
	
	
	for (map <int, int>::iterator itf = hist.begin(); itf!=hist.end(); itf++)		
		out<<itf->first<<"  "<<itf->second<<endl;
		
	return 0;

}





template<typename T, template<typename> class C>
int shuffle_s(C<T> &sq) {
	
	
	int siz=sq.size();
	if(siz==0)
		return -1;
	
	for (int i=0; i<sq.size(); i++) {
		
		int random_pos=irand(siz-1);
	
		T random_card_=sq[random_pos];
	
		sq[random_pos]=sq[siz-1];
		sq[siz-1]=random_card_;
		siz--;
		
	
	}
	
	
	return 0;
	
	
}




void statement() {
	
	cerr<<"\nTo run the program type \n./benchmark (-sup) (-inf)\n\n";
	cerr<<"Only one option can be used at one time."<<endl;
	cerr<<"The program needs some parameters that can be set editing a file called \"parameters.dat\"."<<endl;
	cerr<<"Please read \"ReadMe.txt\" for all the details."<<endl;


}


int benchmark(bool excess, bool defect) {	
	
	
	{	// it erases eventual previous files and prints a warning which will be overwritten if the program doesn't crash
		
		ofstream out1("network.dat");
		ofstream out2("community.dat");
		ofstream statout("statistics.dat");
		out1<<"something went wrong"<<endl;
		out2<<"something went wrong"<<endl;
		statout<<"something went wrong"<<endl;

	}
	
	//	******************* PARAMETERS -----------------------------
	
	ifstream in("parameters.dat");
	
	int num_nodes=unlikely;
	double average_k=unlikely;
	int max_degree=unlikely;
	double tau=unlikely;							// exponent of the degree sequence of the nodes
	double tau2=unlikely;							// exponent for the pdf of the community sizes
	double mixing_parameter=unlikely;
	int nmax=unlikely;
	int nmin=unlikely;
	

	
	bool fixed_range=false;					// this is true only if the user set a fixed range for the community sizes
	
	
	
	cout<<endl<<endl<<endl;
	cout<<"reading parameters.dat..."<<endl;
	
	{	// check 
	
		ifstream check_in("parameters.dat");
		if (!check_in.is_open()) {
			cerr<<"File not found. Where is it?"<<endl;
			statement();
			return -1;
		}
	}
	
	
	
	{
	
	
		ifstream in("parameters.dat");
		streampos sp = in.tellg();
		string word;
		while(in>>word) {
			

			if (word[0]=='#')
				getline(in, word);
			
			
			else {
				
				in.seekg(sp);
				if (num_nodes==unlikely) {
					
					double err;
					in>>err;
					if (fabs(err-int (err))>1e-10) {
						
						cerr<<"ERROR: number of nodes must be an integer"<<endl;
						return -1;
					
					}
				
					num_nodes=int(err);

				}
					
				else if (average_k==unlikely)
					in>>average_k;
				else if  (max_degree==unlikely) {
					
					double err;
					in>>err;
					if (fabs(err-int (err))>1e-10) {
						
						cerr<<"ERROR: the maximum degree must be an integer"<<endl;
						return -1;
					
					}
					
					max_degree=int(err);

				}
				else if  (tau==unlikely)
					in>>tau;
				else if  (tau2==unlikely)
					in>>tau2;	
				else if  (mixing_parameter==unlikely)
					in>>mixing_parameter;
				
				else if  (nmin==unlikely) {
					
					double err;
					in>>err;
					if (fabs(err-int (err))>1e-10) {
						
						cerr<<"ERROR: the minumum community size must be an integer"<<endl;
						return -1;
					
					}
					
					nmin=int(err);

				}
				else if  (nmax==unlikely) {
					
					double err;
					in>>err;
					if (fabs(err-int (err))>1e-10) {
						
						cerr<<"ERROR: the maximum community size must be an integer"<<endl;
						return -1;
					
					}
					
					nmax=int(err);
					fixed_range=true;

				}				
				else
					break;
					
		
			}
			
			sp = in.tellg();
			
		}
		
		
		
		if (num_nodes==unlikely || average_k==unlikely || max_degree==unlikely || tau==unlikely || tau2==unlikely || mixing_parameter==unlikely) {

			cerr<<"ERROR:\tsome parameters are missing"<<endl;
						
			return -1;
		
		}
		
			
		if (num_nodes<=0 || average_k<=0 || max_degree<=0 || mixing_parameter<0 || (nmax<=0 && nmax!=unlikely) || (nmin<=0 && nmin!=unlikely) ) {
		
			cerr<<"ERROR:\tsome positive parameter are negative"<<endl;
			
			return -1;

		
		}
		
		
		
		cout<<"\n**************************************************************"<<endl;
		cout<<"number of nodes:\t"<<num_nodes<<endl;
		cout<<"average degree:\t"<<average_k<<endl;
		cout<<"maximum degree:\t"<<max_degree<<endl;
		cout<<"exponent for the degree distribution:\t"<<tau<<endl;
		cout<<"exponent for the community size distribution:\t"<<tau2<<endl;
		cout<<"mixing parameter:\t"<<mixing_parameter<<endl;
		
		if (fixed_range) {
			cout<<"community size range set equal to ["<<nmin<<" , "<<nmax<<"]"<<endl;
			
			if (nmin>nmax) {
				cerr<<"ERROR: inverted commuity size bounds"<<endl;
				return -1;
			}
			
			if(nmax>num_nodes) {
				cerr<<"ERROR: nmax bigger than the number of nodes"<<endl;
				return -1;
			}
				
			
		
		}
		cout<<"**************************************************************"<<endl<<endl;
		
	
	
	}
	//	******************* PARAMETERS -----------------------------
	
	
	if(excess && defect) {
		
		cerr<<"both options -inf and -sup cannot be used at the same time"<<endl;
		statement();
		return -1;
	
	}

	
	if (mixing_parameter>1) {
		
				
		if(excess || defect)
			cerr<<"Warning: options -sup or -inf cannot be used when the mixing parameter is bigger than 1"<<endl;
		
		
		excess=false;
		defect=false;

	}
		


	
	
	
	// it finds the minimum degree -----------------------------------------------------------------------

	double dmin=solve_dmin(max_degree, average_k, -tau);
	if (dmin==-1)
		return -1;
	
	int min_degree=int(dmin);
	
	
	double media1=integer_average(max_degree, min_degree, tau);
	double media2=integer_average(max_degree, min_degree+1, tau);
	
	if (fabs(media1-average_k)>fabs(media2-average_k))
		min_degree++;
	
	
	
	
	
	
		
	// range for the community sizes
	if (!fixed_range) {
	
		nmax=max_degree;
		nmin=int(min_degree);
		cout<<"-----------------------------------------------------------"<<endl;
		cout<<"community size range automatically set equal to ["<<nmin<<" , "<<nmax<<"]"<<endl;

	}
	
	
	//----------------------------------------------------------------------------------------------------
	
	
	deque <int> degree_seq ;		//  degree sequence of the nodes
	deque <double> cumulative;
	powerlaw(max_degree, min_degree, tau, cumulative);
	
	for (int i=0; i<num_nodes; i++) {
		
		int nn=lower_bound(cumulative.begin(), cumulative.end(), ran4())-cumulative.begin()+min_degree;
		degree_seq.push_back(nn);
	
	}
	
	
	
	//it builds the network
	deque<set<int> > en;
	if (config_model(en, degree_seq, min_degree)==-1)
		return -1;

	//cout<<"done"<<endl;
	
	
	
	for (int i=0; i<en.size(); i++)
		degree_seq[i]=en[i].size();
		
	
	// it assigns the internal degree to each node -------------------------------------------------------------------------
	int max_degree_actual=0;		// maximum internal degree
	int max_degree_actual2=0;		// maximum degree

	deque <int> internal_degree_seq;
	for (int i=0; i<degree_seq.size(); i++) {
		
		double interno=(1-mixing_parameter)*degree_seq[i];
		int int_interno=int(interno);
		
		
		if (ran4()<(interno-int_interno))
			int_interno++;
		
		if (excess) {
			
			while (   (  double(int_interno)/degree_seq[i] < (1-mixing_parameter) )  &&   (int_interno<degree_seq[i])   )
				int_interno++;
				
		
		}
		
		
		if (defect) {
			
			while (   (  double(int_interno)/degree_seq[i] > (1-mixing_parameter) )  &&   (int_interno>0)   )
				int_interno--;
				
		
		}

		
		
		
		internal_degree_seq.push_back(int_interno);
		
		
		if (int_interno>max_degree_actual)
			max_degree_actual=int_interno;
		
		if (degree_seq[i]>max_degree_actual2)
			max_degree_actual2=degree_seq[i];
		
	
	}
	
	
	// it assigns the community size sequence -----------------------------------------------------------------------------
	deque <int> num_seq ;		// degree_seq of the sizes
	
	powerlaw(nmax, nmin, tau2, cumulative);
	
	
	int _num_=0;
	if (!fixed_range && (max_degree_actual+1)>nmin) {
	
		_num_=max_degree_actual+1;			// this helps the assignment of the memberships (it assures that at least one module is big enough to host each node)
		num_seq.push_back(max_degree_actual+1);
	
	}
	
	
	while (true) {
		
		
		int nn=lower_bound(cumulative.begin(), cumulative.end(), ran4())-cumulative.begin()+nmin;
		
		if (nn+_num_<=num_nodes) {
			
			num_seq.push_back(nn);				
			_num_+=nn;
		
		}
		else
			break;
		
		
	}
	
	
	num_seq[num_seq.size()-1]+= num_nodes - _num_;
	int ncom=num_seq.size();
	
	//cout<<"\n----------------------------------------------------------"<<endl;

	/*
	cout<<"community sizes"<<endl;
	for (int i=0; i<num_seq.size(); i++)
		cout<<num_seq[i]<<" ";
	cout<<endl<<endl;
	//*/
	

	deque<deque<int> > member_matrix; // matrix of memberships: row i contains the nodes belonging to community i
	
	deque <int> first;
	for (int i=0; i<ncom; i++)
		member_matrix.push_back(first);
	
	deque <int> refused(degree_seq.size());
	for (int i=0; i<degree_seq.size(); i++)
		refused[i]=i;
		
		
	int k_r=0;
	
	// it decides the memberships
	
	while (refused.size()>0) {
		
		
		k_r++;
		deque<int> new_refused;
		
		for (int i=0; i<refused.size(); i++) {
		
			int random_module=irand(ncom-1);

			if (internal_degree_seq[refused[i]]<(num_seq[random_module])) {
			  
				
				if (member_matrix[random_module].size()==num_seq[random_module]) {
					new_refused.push_back(member_matrix[random_module][0]);
					member_matrix[random_module].pop_front();			
				}
				

				member_matrix[random_module].push_back(refused[i]);
					
					
					
			}
			
			else {
				new_refused.push_back(refused[i]);
			}
		}
		
		refused.clear();
		refused=new_refused;
		int missing_links=0;
		for (int j=0; j<refused.size(); j++)
			missing_links+=internal_degree_seq[refused[j]];
		
		
		
		if (k_r>3*num_nodes) {
		
			k_r=0;
			cout<<"it took too long to decide the memberships; I will try to change the community sizes"<<endl;
			ncom--;
			member_matrix.pop_back();
			for (int im=0; im<member_matrix.size(); im++)
				member_matrix[im].clear();
			
			refused.clear();
			for (int im=0; im<degree_seq.size(); im++)
				refused.push_back(im);

			
			
			if (ncom==0) {
				
				cerr<<"it did not work, sorry"<<endl;
				return -1;

			}
			
			change_community_size(num_seq);
			
			cout<<"new community sizes"<<endl;
			for (int i=0; i<num_seq.size(); i++)
				cout<<num_seq[i]<<" ";
			cout<<endl<<endl;

			
		}
			
		
	
	}
	
	/*
	for (int i=0; i<member_matrix.size(); i++) {
		for (int j=0; j<member_matrix[i].size(); j++)
			cout<<member_matrix[i][j]<<" ";
		cout<<endl;
		}
	*/	
	
	if (ncom==1 && mixing_parameter<1) {
		
		cerr<<"ERROR: this program needs more than one community to work fine"<<endl;
		return -1;
	
	}
	//cout<<"\n----------------------------------------------------------"<<endl;

	
	
	
	// ------------------------ this is done to check if the sum of the internal degree is an even number. if not, the program will change it in such a way to assure that. 
	
	
			
	for (int i=0; i<member_matrix.size(); i++) {
	
		
		int internal_cluster=0;
		for (int j=0; j<member_matrix[i].size(); j++)
			internal_cluster+=internal_degree_seq[member_matrix[i][j]];
		
		//cout<<"here "<<member_matrix.size()<<" "<<internal_cluster<<endl;
		
		if(internal_cluster % 2 != 0) {
			
			
			//cout<<"correction for even kin"<<endl;
			bool default_flag=false;
			
			if(excess)
				default_flag=true;
			else if(defect)
				default_flag=false;
			else if (ran4()>0.5)
				default_flag=true;
			
			if(default_flag) {
				
				
				for (int j=0; j<member_matrix[i].size(); j++) {
					if (    (internal_degree_seq[member_matrix[i][j]]<member_matrix[i].size()-1) && (internal_degree_seq[member_matrix[i][j]] < degree_seq[member_matrix[i][j]]) ) {
						
						internal_degree_seq[member_matrix[i][j]]++;
						break;
						
					}
				
				}
			
			
			}
			
			
			else {
				
				
				for (int j=0; j<member_matrix[i].size(); j++) {
					if (internal_degree_seq[member_matrix[i][j]] > 0 ) {
						
						internal_degree_seq[member_matrix[i][j]]--;
						break;
						
					}
				
				}
			
			
			}
		
		
		}
	
	
	}
	
	// ------------------------ this is done to check if the sum of the internal degree is an even number. if not, the program will change it in such a way to assure that. 
	
	
	
	
	
	deque <int> member_list(num_nodes);
	for (int i=0; i<member_matrix.size(); i++)
		for (int j=0; j<member_matrix[i].size(); j++)
			member_list[member_matrix[i][j]]=i;

	
	deque<int>internal_plusminus_seq(degree_seq.size());
	
	int var2=0;
	for (int i=0; i<num_nodes; i++) {
			
			int inte=0;
			for (set<int>::iterator it=en[i].begin(); it!=en[i].end(); it++)
				if (member_list[*it]==member_list[i])
					inte++;
			
			internal_plusminus_seq[i]=inte-internal_degree_seq[i];
			var2+=(internal_plusminus_seq[i])*(internal_plusminus_seq[i]);
	
	}
	
	
		
	
	
	// -----------------------------------------------------
	
	
	
	
	int best_var=var2;
	int max_counter=num_nodes / member_matrix.size();		// maximum number of iterations (increasing this parameter would make the program more accurate and slower)
	//cout<<"max_counter : "<<max_counter<<endl;
	int counter_=0;
	
	if (mixing_parameter>1) {
		cout<<"Warning: the mixing parameter is > 1!\nI skip the rewiring process and you will get a random network"<<endl;
		counter_=max_counter;
	}
	
	//else
	//	cout<<"Rewiring process: this may take a while......"<<endl;
	
	int deqar_size=0;
	for (int i=0; i<member_matrix.size(); i++)
		if(member_matrix[i].size()>deqar_size)
			deqar_size=member_matrix[i].size();
	
	if (max_degree_actual2>deqar_size)
		deqar_size=max_degree_actual2;
	
	
	//cout<<"deqar_size "<<deqar_size<<endl;
	deque<int> deqar(deqar_size);
	int deq_s;
	
	
	
	
	while (counter_<max_counter) {
		
		counter_++;
		int stopper2_limit=3;
		
		for (int i=0; i<member_matrix.size(); i++) {
			
			for (int j=0; j<member_matrix[i].size(); j++) {
				
				int nodo=member_matrix[i][j];
				
				int stopper2=0;
				while (stopper2<stopper2_limit && internal_plusminus_seq[nodo]<0) {
				
				//---------------------------------------------------------------------------------
					
					stopper2++;
					
					
					
					deq_s=0;		// not neighbors of nodo with same membership
					for (int k=0; k<member_matrix[member_list[nodo]].size(); k++) {
							
						int candidate=member_matrix[member_list[nodo]][k];
						if ( (en[nodo].find(candidate)==en[nodo].end() ) && (candidate!=nodo))
							deqar[deq_s++]=candidate;
						
					}
					
					if(deq_s==0)
						break;
					
					int random_mate=deqar[irand(deq_s-1)];
					
					
					
					
					
					{
					
						
						deq_s=0;		// neighbors of nodo with different membership
						for (set<int>::iterator it_est=en[nodo].begin(); it_est!=en[nodo].end(); it_est++)
							if (member_list[*it_est]!=member_list[nodo])
								deqar[deq_s++]=*it_est;
						
						
						if(deq_s==0)
							break;
						
						int old_node=deqar[irand(deq_s-1)];

						
						deq_s=0;	// neighbors of random_mate which are not old_node's neighbors
						for (set<int>::iterator it_est=en[random_mate].begin(); it_est!=en[random_mate].end(); it_est++)
							if ((old_node!=(*it_est)) && (en[old_node].find(*it_est)==en[old_node].end()))
								deqar[deq_s++]=*it_est;
						
						
						
						if(deq_s==0)
							break;

						int nodo_h=deqar[irand(deq_s-1)];
						
						
						
						
						int nodes_here[4];
						
						nodes_here[0]= nodo;
						nodes_here[1]= random_mate;
						nodes_here[2]= old_node;
						nodes_here[3]= nodo_h;
						
						
						int current_plus_minus[4];
						for (int k=0; k<4; k++)
							current_plus_minus[k]=internal_plusminus_seq[nodes_here[k]];
						
						
						current_plus_minus[0]++;
						current_plus_minus[1]++;
						
						if (member_list[nodo_h]==member_list[random_mate]) {
							current_plus_minus[1]--;
							current_plus_minus[3]--;
						}
						
						
						
						if (member_list[nodo_h]==member_list[old_node]) {
							current_plus_minus[2]++;
							current_plus_minus[3]++;
						}
						
						
						
						int impr=0;
						for (int k=0; k<4; k++)
							impr+= (internal_plusminus_seq[nodes_here[k]])*(internal_plusminus_seq[nodes_here[k]]) - (current_plus_minus[k])*(current_plus_minus[k]);
						
						if(impr<0)
							break;
						
						
						//if(impr>0)
						//	cout<<"impr "<<impr<<endl;
						
						
						
						var2-=impr;
						
						en[nodo].erase(old_node);
						en[nodo].insert(random_mate);
						
						en[old_node].erase(nodo);
						en[old_node].insert(nodo_h);

						en[random_mate].erase(nodo_h);
						en[random_mate].insert(nodo);
						
						en[nodo_h].erase(random_mate);
						en[nodo_h].insert(old_node);
						
						
						
						for (int k=0; k<4; k++)
							internal_plusminus_seq[nodes_here[k]]=current_plus_minus[k];
						
					
					}
					
					

				
				//---------------------------------------------------------------------------------

				
				}
				
				
				
				while (stopper2<stopper2_limit && internal_plusminus_seq[nodo]>0) {
				
				//---------------------------------------------------------------------------------
					
					stopper2++;
					
					
					
					int random_module=irand(member_matrix.size()-1);
					while (random_module==i)
						random_module=irand(member_matrix.size()-1);
						
					int random_mate=member_matrix[random_module][irand(member_matrix[random_module].size()-1)];
					
					
					
					
					
					if ( en[nodo].find(random_mate)==en[nodo].end() ) {
					
						
						
						
						
						deq_s=0;		// neighbors of nodo with same membership
						for (set<int>::iterator it_est=en[nodo].begin(); it_est!=en[nodo].end(); it_est++)
							if (member_list[*it_est]==member_list[nodo])
								deqar[deq_s++]=*it_est;
						
						
						if(deq_s==0)
							break;
						
						int old_node=deqar[irand(deq_s-1)];

						
						deq_s=0;		// neighbors of random_mate which are not old_node's neighbors
						for (set<int>::iterator it_est=en[random_mate].begin(); it_est!=en[random_mate].end(); it_est++)
							if ((old_node!=(*it_est)) && (en[old_node].find(*it_est)==en[old_node].end()))
								deqar[deq_s++]=*it_est;
						
						
						
						if(deq_s==0)
							break;

						int nodo_h=deqar[irand(deq_s-1)];
						
						
						
						
						
						int nodes_here[4];
						
						nodes_here[0]= nodo;
						nodes_here[1]= random_mate;
						nodes_here[2]= old_node;
						nodes_here[3]= nodo_h;
						
						
						
												
						int current_plus_minus[4];
						for (int k=0; k<4; k++)
							current_plus_minus[k]=internal_plusminus_seq[nodes_here[k]];

						
						
						current_plus_minus[0]--;
						current_plus_minus[2]--;
						
						if (member_list[nodo_h]==member_list[random_mate]) {
							current_plus_minus[1]--;
							current_plus_minus[3]--;
						}
						
						
						
						if (member_list[nodo_h]==member_list[old_node]) {
							current_plus_minus[2]++;
							current_plus_minus[3]++;
						}
						
						
						
						int impr=0;
						for (int k=0; k<4; k++)
							impr+= (internal_plusminus_seq[nodes_here[k]])*(internal_plusminus_seq[nodes_here[k]]) - (current_plus_minus[k])*(current_plus_minus[k]);
						
						
						if(impr<0)
							break;
						
						
						
						//if(impr>0)
						//	cout<<"impr "<<impr<<endl;
						
						
						
						var2-=impr;
						
						en[nodo].erase(old_node);
						en[nodo].insert(random_mate);
						
						en[old_node].erase(nodo);
						en[old_node].insert(nodo_h);

						en[random_mate].erase(nodo_h);
						en[random_mate].insert(nodo);
						
						en[nodo_h].erase(random_mate);
						en[nodo_h].insert(old_node);
						
						
						for (int k=0; k<4; k++)
							internal_plusminus_seq[nodes_here[k]]=current_plus_minus[k];
												
						
						
						
					
					}
					
					

				
				//---------------------------------------------------------------------------------

				
				}
				
			
			
			}
		
		
		}
		
		//if (double(best_var - var2)/best_var > 0)
			//cout<<"rewiring... diff: "<<var2<<"\t\t(relative improvement: "<<double(best_var - var2)/best_var<<")"<<endl;
		
		if (var2==0)
			break;
		
		if (best_var>var2) {
			
			best_var=var2;
			counter_=0;
			
		}
		

		
		
	}
	
	
	
	
	
	//------------------------------------ CHECK  ---------------------------------------------------------------
	
		
	/*
		
	{
		
		int positive_nodes=0;
		int negative_nodes=0;
		int equal_nodes=0;
		
		deque<int> pos_n;
		deque<int> neg_n;
		
		for (int u=0; u<en.size(); u++) {
			
			int internal_check=0;
			
			for (set<int>::iterator it=en[u].begin(); it!=en[u].end(); it++)
				if(member_list[*it]==member_list[u])
					internal_check++;
			
			
			int plus_minus=internal_plusminus_seq[u];
			
			if(plus_minus>0) {
				positive_nodes++;
				//pos_n.push_back(member_list[u]);			
				pos_n.push_back(u);			
			}
			
			if(plus_minus<0) {
				negative_nodes++;
				//neg_n.push_back(member_list[u]);
				neg_n.push_back(u);

			}
			
			if(plus_minus==0)
				equal_nodes++;
			
			
			
			if(internal_plusminus_seq[u] + internal_degree_seq[u]!=internal_check) {
				
				cerr<<"error a"<<endl;
				int a;
				cin>>a;
			
			}
			
			
			if(degree_seq[u]!=en[u].size() ) {
				
				cerr<<"error b"<<endl;
				int a;
				cin>>a;
			
			}
			
			
		
		}
		
		cout<<"pos: "<<double(positive_nodes)/(en.size())<<";\tneg: "<<double(negative_nodes)/(en.size())<<";\teq: "<<double(equal_nodes)/(en.size())<<endl;
		
		
		//*
		cout<<"pos"<<endl;
		for (int k=0; k<pos_n.size(); k++)
			cout<<pos_n[k]<<"\t";
		cout<<endl;
		
		cout<<"neg"<<endl;
		for (int k=0; k<neg_n.size(); k++)
			cout<<neg_n[k]<<"\t";
		cout<<endl;
		
		neg_n.clear();
		pos_n.clear();
		
		//* /
		
		
	}
	
	//*/
	
	
	//------------------------------------ CHECK  ---------------------------------------------------------------


	
	//------------------------------------ Erasing links   ------------------------------------------------------
	
	int eras_add_times=0;
	
	if (excess) {
		
		for (int i=0; i<num_nodes; i++)
			while ( (degree_seq[i]>1) &&  double(internal_plusminus_seq[i] + internal_degree_seq[i])/degree_seq[i] < 1 - mixing_parameter) {
			
			//---------------------------------------------------------------------------------
				
				int nodo=i;
				
				cout<<"degree sequence changed to respect the option -sup ... "<<++eras_add_times<<endl;
				
				deq_s=0;		// neighbors of nodo with different membership
				for (set<int>::iterator it_est=en[nodo].begin(); it_est!=en[nodo].end(); it_est++)
					if (member_list[*it_est]!=member_list[nodo])
						deqar[deq_s++]=*it_est;
				
				
				if(deq_s==en[nodo].size()) {	// this shouldn't happen...
				
					cerr<<"sorry, something went wrong: there is a node which does not respect the constraints."<<endl;
					return -1;
				
				}
				
				int random_mate=deqar[irand(deq_s-1)];
				
				en[nodo].erase(random_mate);
				en[random_mate].erase(nodo);
				
				degree_seq[i]--;
				degree_seq[random_mate]--;
				
			
		
		
			}
			
	
	}
	
	
	
	if (defect) {
			
		for (int i=0; i<num_nodes; i++)
			while ( (degree_seq[i]<num_nodes) &&  double(internal_plusminus_seq[i] + internal_degree_seq[i])/degree_seq[i] > 1 - mixing_parameter) {
				
				//---------------------------------------------------------------------------------
					
				int nodo=i;
				
				cout<<"degree sequence changed to respect the option -inf ... "<<++eras_add_times<<endl;


				int stopper_here=num_nodes;
				int stopper_=0;
				
				int random_mate=irand(num_nodes-1);
				while ( !(member_list[random_mate]!=member_list[nodo] && en[nodo].find(random_mate)==en[nodo].end()) && (stopper_<stopper_here) ) {
					
					random_mate=irand(num_nodes-1);
					stopper_++;
				
				
				}
				
				if(stopper_==stopper_here) {	// this shouldn't happen...
				
					cerr<<"sorry, something went wrong: there is a node which does not respect the constraints."<<endl;
					return -1;
				
				}
				
				
				
				en[nodo].insert(random_mate);
				en[random_mate].insert(nodo);
				
				degree_seq[i]++;
				degree_seq[random_mate]++;
								
		
			}
			
		
	}

	//------------------------------------ Erasing links   ------------------------------------------------------

	
	
	
	//------------------------------------ CHECK  ---------------------------------------------------------------
	
	
		
	/*
		
	{
		
		int positive_nodes=0;
		int negative_nodes=0;
		int equal_nodes=0;
		
		deque<int> pos_n;
		deque<int> neg_n;
		
		for (int u=0; u<en.size(); u++) {
			
			int internal_check=0;
			
			for (set<int>::iterator it=en[u].begin(); it!=en[u].end(); it++)
				if(member_list[*it]==member_list[u])
					internal_check++;
			
			
			int plus_minus=internal_plusminus_seq[u];
			
			if(plus_minus>0) {
				positive_nodes++;
				pos_n.push_back(member_list[u]);			
			}
			
			if(plus_minus<0) {
				negative_nodes++;
				neg_n.push_back(member_list[u]);

			}
			
			if(plus_minus==0)
				equal_nodes++;
			
			
			
			if(internal_plusminus_seq[u] + internal_degree_seq[u]!=internal_check) {
				
				cerr<<"error a "<< u <<endl;
				int a;
				cin>>a;
			
			}
			
			
			if(degree_seq[u]!=en[u].size() ) {
				
				cerr<<"error b"<<endl;
				int a;
				cin>>a;
			
			}
			
			
		
		}
		
		cout<<"pos: "<<double(positive_nodes)/(en.size())<<";\tneg: "<<double(negative_nodes)/(en.size())<<";\teq: "<<double(equal_nodes)/(en.size())<<endl;
		
		
		//*
		cout<<"pos"<<endl;
		for (int k=0; k<pos_n.size(); k++)
			cout<<pos_n[k]<<"\t";
		cout<<endl;
		
		cout<<"neg"<<endl;
		for (int k=0; k<neg_n.size(); k++)
			cout<<neg_n[k]<<"\t";
		cout<<endl;
		
		neg_n.clear();
		pos_n.clear();
		
		//* /
		
		
	}
	
	//*/

	
	
	
	
	//------------------------------------ CHECK  ---------------------------------------------------------------



	double ratio_true=0;
	double variance=0;
	
	
	//cout<<"done"<<endl;
	
	deque <double> double_mixing;
	for (int i=0; i<degree_seq.size(); i++) {
					
		double_mixing.push_back(1.-double(internal_plusminus_seq[i] + internal_degree_seq[i])/degree_seq[i]);
		ratio_true+=double(internal_plusminus_seq[i] + internal_degree_seq[i])/degree_seq[i];
		variance+=(double(internal_plusminus_seq[i] + internal_degree_seq[i])/degree_seq[i])*(double(internal_plusminus_seq[i] + internal_degree_seq[i])/degree_seq[i]);
	}
	//cout<<"\n----------------------------------------------------------"<<endl;
	//cout<<endl;
	
	ratio_true=ratio_true/degree_seq.size();
	variance=variance/degree_seq.size();
	variance -= ratio_true*ratio_true;
	
	double density=0; 
	double sparsity=0;
	int edges=0;
	
	for (int i=0; i<member_matrix.size(); i++) {

		double media_int=0;
		double media_est=0;
		
		for (int j=0; j<member_matrix[i].size(); j++) {
			
			media_int+=internal_plusminus_seq[member_matrix[i][j]] + internal_degree_seq[member_matrix[i][j]];
			media_est+=degree_seq[member_matrix[i][j]]- internal_plusminus_seq[member_matrix[i][j]] - internal_degree_seq[member_matrix[i][j]];
			edges+=degree_seq[member_matrix[i][j]];
		
		}
		
		double pair_num=(member_matrix[i].size()*(member_matrix[i].size()-1));
		double pair_num_e=((num_nodes-member_matrix[i].size())*(member_matrix[i].size()));
		
		density+=media_int/pair_num;
		sparsity+=media_est/pair_num_e;
		
		
	
	}
	
	density=density/member_matrix.size();
	sparsity=sparsity/member_matrix.size();
	
	
	


	ofstream out1("network.dat");
	for (int u=0; u<en.size(); u++) {

		set<int>::iterator it=en[u].begin();
	
		int uu=0;
		while (it!=en[u].end())
			out1<<u+1<<"\t"<<*(it++)+1<<endl;

	}
		

	
	ofstream out2("community.dat");

	for (int i=0; i<member_list.size(); i++)
		out2<<i+1<<"\t"<<member_list[i]+1<<endl;
	

	cout<<"-----------------------------------------------------------"<<endl;

	cout<<"network of "<<num_nodes<<" vertices and "<<edges/2<<" edges"<<";\t average degree = "<<double(edges)/num_nodes<<endl;
	cout<<"average mixing parameter: "<< 1 - ratio_true<<" +/- "<<sqrt(variance)<<endl;
	cout<<"p_in: "<<density<<"\tp_out: "<<sparsity<<endl;
	
	
	
	ofstream statout("statistics.dat");
	
	statout<<"degree distribution (degree-occurrences)"<<endl;
	log_histogram(degree_seq, statout, 10);
	statout<<endl<<"--------------------------------------"<<endl;
	
		
	statout<<"community distribution (size-occurrences)"<<endl;
	histogram(num_seq, statout);
	statout<<endl<<"--------------------------------------"<<endl;

	statout<<"mixing parameter"<<endl;
	not_norm_histogram(double_mixing, statout, 20, 0, 1.1);

	
	cout<<endl<<endl;
	
	
			
	
	return 0;
	
}





void srand_file(void) {

	ifstream in("bench_seed.dat");
	if (!in.is_open()) {
		srand4();
		seed=irand(R2_IM2);
	}
	else
		in>>seed;
	
	if (seed < 1 || seed>R2_IM2)
		seed=1;
	
	ofstream out("bench_seed.dat");
	out<<seed+1<<endl;
	

}




int main(int argc, char * argv[]) {
	
		
	srand_file();
	
	
	bool weighted=false;
	bool value=false;
	
	
	
	// parse command line ----------------------------------------------------------------------------------
	
	string s;
	bool excess=false;
	bool defect=false;
	
	
	int _arg_ = 1;	
	while (argc > _arg_) {
		
	
		s = argv[_arg_];
		
		if (s== "-sup") {
			
			excess=true;
		}
	
		else if (s== "-inf") {

			defect=true;
		}
	
	
	
		else {
		
			cerr<<"ERROR"<<endl<<endl;
			statement();
			return 0;
		}
		
		_arg_++;
	
	}	
	
	// parse command line ----------------------------------------------------------------------------------

	

	
	benchmark(excess, defect);
	
	return 0;
	
}


