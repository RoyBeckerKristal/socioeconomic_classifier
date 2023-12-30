import numpy
import sys
import pandas
import matplotlib.pyplot as plt
from bidi import algorithm as bidialg

# Exploring data    
def explore_data(kalpi_data, party_lists, create_graphs=False):
    # Obtaining different sections of the data:
    # Clustered vs. unclustered kalpi data
    clustered_kalpi_data = kalpi_data[kalpi_data["cluster"].notna()]
    unclustered_kalpi_data = kalpi_data[kalpi_data["cluster"].isna()]
    # Kalpi data for each sector
    jewish_kalpi_data = kalpi_data.loc[kalpi_data["jew"]==1].loc[kalpi_data["pal"]==0]
    arab_kalpi_data = kalpi_data.loc[kalpi_data["jew"]==0].loc[kalpi_data["pal"]==1]
    mixed_kalpi_data = kalpi_data.loc[kalpi_data["jew"]==1].loc[kalpi_data["pal"]==1]
    druze_kalpi_data = kalpi_data.loc[kalpi_data["jew"]==0].loc[kalpi_data["pal"]==0]
    # Clustered kalpi data for each sector
    jewish_clustered_kalpi_data = jewish_kalpi_data[jewish_kalpi_data["cluster"].notna()].copy()
    arab_clustered_kalpi_data = arab_kalpi_data[arab_kalpi_data["cluster"].notna()].copy()
    mixed_clustered_kalpi_data = mixed_kalpi_data[mixed_kalpi_data["cluster"].notna()].copy()
    druze_clustered_kalpi_data = druze_kalpi_data[druze_kalpi_data["cluster"].notna()].copy()

    # Party titles and colors for graphs:
    party_titles = [pl["title"] for pl in party_lists.values()]
    party_colors = [pl["color"] for pl in party_lists.values()]

    # Printing general statistics
    print ("Kalpi data:")
    print ("Sector", "Total", "Clustered", "%", sep="\t")
    print ("All", kalpi_data.shape[0], clustered_kalpi_data.shape[0], round(100*clustered_kalpi_data.shape[0]/kalpi_data.shape[0],2), sep="\t")
    print ("Jewish", jewish_kalpi_data.shape[0], jewish_clustered_kalpi_data.shape[0], round(100*jewish_clustered_kalpi_data.shape[0]/jewish_kalpi_data.shape[0],2), sep="\t")
    print ("Arab", arab_kalpi_data.shape[0], arab_clustered_kalpi_data.shape[0], round(100*arab_clustered_kalpi_data.shape[0]/arab_kalpi_data.shape[0],2), sep="\t")
    print ("Mixed", mixed_kalpi_data.shape[0], mixed_clustered_kalpi_data.shape[0], round(100*mixed_clustered_kalpi_data.shape[0]/mixed_kalpi_data.shape[0],2), sep="\t")
    print ("Druze", druze_kalpi_data.shape[0], druze_clustered_kalpi_data.shape[0], round(100*druze_clustered_kalpi_data.shape[0]/druze_kalpi_data.shape[0],2), sep="\t")

    # Printing vote distribution in clustered vs. unclustered kalpiot
    clustered_party_means = clustered_kalpi_data[list(party_lists.keys())].mean()
    unclustered_party_means = unclustered_kalpi_data[list(party_lists.keys())].mean()
    print ("\nVoting distribution of SE-indexed kalpiot:")
    print(clustered_party_means)
    print ("\nVoting distribution of unindexed kalpiot:")
    print(unclustered_party_means)

    # Creating graph for vote distribution in clustered vs. unclustered kalpiot
    if create_graphs:
        try:
            # Setting bar values
            clustered_values = clustered_party_means.to_numpy()*100
            unclustered_values = unclustered_party_means.to_numpy()*100
            # Setting x and y tick values
            x = numpy.arange(1,len(clustered_values)+1)
            max_value = (clustered_values.max()//5)*5+5 if clustered_values.max()>unclustered_values.max() else (unclustered_values.max()//5)*5+5
            y = list(range(0, int(max_value+5), 5))
            # Creating figure and chart area
            fig = plt.figure(figsize=(6,1.25))
            ax = fig.add_axes([0.1,0.15,0.8,0.6])
            # Setting chart details:
            # Bars
            ax.bar(x-0.2, clustered_values, width=0.4, color='blue')
            ax.bar(x+0.2, unclustered_values, width=0.4, color='orange')
            # Ticks
            ax.set_xticks(x)
            ax.set_yticks(y)
            ax.tick_params(axis='x',length=0)
            # Tick labels (including proper display for Hebrew)
            ax.set_xticklabels(list(map(lambda s: bidialg.get_display(s), party_titles)), fontsize=9)
            ax.set_yticklabels(y, fontsize=9)
            # Grid lines
            ax.grid(axis='y')
            ax.set_axisbelow(True)
            # Title
            ax.set_title("Ballots(%) in SE-indexed(blue) vs. unindexed(orange) kalpiot", fontsize=10)
            # Displaying the figure
            plt.show()
            print ("Graph printed")
        except:
            print ("Problem printing clustered vs. unclustered votes graph.")

    # Printing indexed kalpi distribution by sector and index
    print ("\nClustered kalpiot by sector and SE-index:")
    print ("SE-index", "All", "Jewish", "Arab", "Mixed", "Druze", sep="\t")
    kalpiot_by_cluster = numpy.zeros(10, dtype=int)
    jewish_kalpiot_by_cluster = numpy.zeros(10, dtype=int)
    arab_kalpiot_by_cluster = numpy.zeros(10, dtype=int)
    mixed_kalpiot_by_cluster = numpy.zeros(10, dtype=int)
    druze_kalpiot_by_cluster = numpy.zeros(10, dtype=int)

    for c in range(10):
        kalpiot_by_cluster[c] = clustered_kalpi_data[clustered_kalpi_data["cluster"]==c].shape[0]
        jewish_kalpiot_by_cluster[c] = jewish_clustered_kalpi_data[jewish_clustered_kalpi_data["cluster"]==c].shape[0]
        arab_kalpiot_by_cluster[c] = arab_clustered_kalpi_data[arab_clustered_kalpi_data["cluster"]==c].shape[0]
        mixed_kalpiot_by_cluster[c] = mixed_clustered_kalpi_data[mixed_clustered_kalpi_data["cluster"]==c].shape[0]
        druze_kalpiot_by_cluster[c] = druze_clustered_kalpi_data[druze_clustered_kalpi_data["cluster"]==c].shape[0]
        print(c+1, kalpiot_by_cluster[c], jewish_kalpiot_by_cluster[c], arab_kalpiot_by_cluster[c], mixed_kalpiot_by_cluster[c], druze_kalpiot_by_cluster[c], sep="\t")
        

    # Creating graph for clustered kalpiot by sector and index
    if create_graphs:
        try:
            # Creating figure and chart area
            fig,ax = plt.subplots(1,5,figsize=(8.0,1.4))
            fig.tight_layout()
            # Setting figure title
            fig.suptitle("Distribution of kalpiot by sector and SE-index", fontsize=11)

            # Setting x tick values
            x=numpy.arange(10)
            # Setting data for each chart
            graph_data = [kalpiot_by_cluster, jewish_kalpiot_by_cluster, arab_kalpiot_by_cluster, mixed_kalpiot_by_cluster, druze_kalpiot_by_cluster]
            # Title and bar colors for each chart
            graph_titles = ["All", "Jewish", "Arab", "Mixed", "Druze"]
            graph_colors = ["black", "blue", "green", "purple", "red"]
            # For each chart:
            for i in range(5):
                # Setting appropriate y tick values based on maximal y value
                max_value = (graph_data[i].max()//10)*10+10 if graph_data[i].max() < 80 else (graph_data[i].max()//50)*50+50
                if max_value < 40:
                    y_step = 5
                elif max_value < 90:
                    y_step = 10
                else:
                    y_step = max_value // 5
                y = list(range(0,max_value+y_step,y_step))
                # Positioning chart on chart area
                ax[i].set_position([0.05+0.195*i, 0.15, 0.14, 0.52])
                # Setting chart details:
                # Bars
                ax[i].bar(x,graph_data[i],color=graph_colors[i])
                # Chart title
                ax[i].set_title(graph_titles[i],fontsize=10)
                # Ticks and tick labels
                ax[i].set_xticks(x)
                ax[i].set_xticklabels(x+1,fontsize=8)
                ax[i].set_yticks(y)
                ax[i].set_yticklabels(y,fontsize=8)
                # Grid lines
                ax[i].grid(axis="y")
                ax[i].set_axisbelow(True)

            # Displaying the figure
            plt.show()
            print ("Graph printed")
        except:
            print ("Problem printing clustered kalpiot by sector and index graph.")


    # Printing party list data by index
    print("\nParty list mean results (%) by index:")
    print("Index\t","Legal",sep="\t",end="\t")
    for p_id, p_info in party_lists.items():
        print(p_info["title"],end="\t")
    print("\nunindexed",round(unclustered_kalpi_data["prop_voters"].mean()*100,2), sep="\t", end="\t")
    for upm in unclustered_party_means:
        print(round(upm*100,2), end="\t")
    print("\nindexed\t",round(clustered_kalpi_data["prop_voters"].mean()*100,2), sep="\t", end="\t")
    for cpm in clustered_party_means:
        print(round(cpm*100,2), end="\t")
    print()
    for i in range(10):
        print(str(i+1)+"\t", round(clustered_kalpi_data[clustered_kalpi_data["cluster"]==i]["prop_voters"].mean()*100,2), sep="\t", end="\t")
        for pl in party_lists.keys():
            print(round(clustered_kalpi_data[clustered_kalpi_data["cluster"]==i][pl].mean()*100,2), end="\t")
        print()

    # Creating graph for sector effect on party votes for SE-index 5
    if create_graphs:
        try:
            # Setting bar values
            jewish_index_5_means = jewish_clustered_kalpi_data[jewish_clustered_kalpi_data["cluster"]==4][list(party_lists.keys())].mean()*100
            arab_index_5_means = arab_clustered_kalpi_data[arab_clustered_kalpi_data["cluster"]==4][list(party_lists.keys())].mean()*100
            mixed_index_5_means = mixed_clustered_kalpi_data[mixed_clustered_kalpi_data["cluster"]==4][list(party_lists.keys())].mean()*100
            druze_index_5_means = druze_clustered_kalpi_data[druze_clustered_kalpi_data["cluster"]==4][list(party_lists.keys())].mean()*100
            # Setting x and y tick values
            x = numpy.arange(1, len(jewish_index_5_means)+1)
            max_value = (numpy.array([jewish_index_5_means.max(), arab_index_5_means.max(), mixed_index_5_means.max(), druze_index_5_means.max()]).max() // 10) * 10 + 10
            y = range(0, int(max_value)+10, 10)
            # Creating figure and chart area
            fig = plt.figure(figsize=(7,1.5))
            ax = fig.add_axes([0.05,0.13,0.9,0.7])
            # Setting chart details:
            # Bars
            ax.bar(x-0.225, jewish_index_5_means, width=0.15, color='blue')
            ax.bar(x-0.075, mixed_index_5_means, width=0.15, color='purple')
            ax.bar(x+0.075, arab_index_5_means, width=0.15, color='green')
            ax.bar(x+0.225, druze_index_5_means, width=0.15, color='red')
            # Ticks
            ax.set_xticks(x)
            ax.set_yticks(y)
            ax.tick_params(axis='x',length=0)
            # Tick labels (including proper display for Hebrew)
            ax.set_xticklabels(list(map(lambda s: bidialg.get_display(s), party_titles)), fontsize=9)
            ax.set_yticklabels(y, fontsize=8)
            # Grid lines
            ax.grid(axis='y')
            ax.set_axisbelow(True)
            # Title
            ax.set_title("Ballots(%) in Jewish(blue), Mixed(purple), Arab(green) and Druze etc.(red) kalpiot with SE-index 5", fontsize=9)
            # Displaying the figure
            plt.show()
            print ("Graph printed")
        except:
            print("Problem printing sector effect graph.")
    

    # Printing results by index in Jewish kalpiot
    jewish_clustered_kalpi_data["arab_bloc"] = jewish_clustered_kalpi_data["WM"] + jewish_clustered_kalpi_data["D"] + jewish_clustered_kalpi_data["AM"]
    jewish_clustered_kalpi_data["right_bloc"] = jewish_clustered_kalpi_data["MHL"] + jewish_clustered_kalpi_data["T"] + jewish_clustered_kalpi_data["B"]
    jewish_clustered_kalpi_data["left_bloc"] = jewish_clustered_kalpi_data["PH"] + jewish_clustered_kalpi_data["KN"] +\
                                               jewish_clustered_kalpi_data["AMT"] + jewish_clustered_kalpi_data["MRC"]
    jewish_clustered_kalpi_data["haredi_bloc"] = jewish_clustered_kalpi_data["G"] + jewish_clustered_kalpi_data["SS"]

    jewish_means_by_index = jewish_clustered_kalpi_data[["cluster","prop_voters","AMT","B","G","T","KN","L","MHL","MRC","PH","SS","others",\
                                                        "arab_bloc","right_bloc","left_bloc","haredi_bloc"]].groupby(["cluster"]).mean().reset_index().sort_values(by=["cluster"])
    print("\nJewish results by index:")
    print("Index","Legal",sep="\t",end="\t")
    for c in list(jewish_means_by_index)[2:]:
        title = party_lists[c]["title"] if c in party_lists.keys() else c
        print(title, end="\t")
    print()
    for i in range(10):
        print(i+1, (jewish_means_by_index.iloc[[i],1:]*100).round(2).to_string(index=False,header=False), sep="\t")

    # Creating graph for Jewish votes by index
    if create_graphs:
        try:
            # Data for the charts
            chart_data = jewish_means_by_index[["PH","KN","AMT","MRC","MHL","T","B","L","G","SS","arab_bloc","others"]]
            # Configuring figure, chart area and figure title
            fig, ax = plt.subplots(3,4,figsize=(5.5,3.5))
            fig.tight_layout()
            fig.suptitle("Party list mean results (%) by SE-index in Jewish sector", fontsize=11)

            # Iterating through charts
            for c in range (chart_data.shape[1]):
                # Setting x and y tick values
                x = numpy.arange(10)
                y_values = chart_data.iloc[:,c].to_numpy()*100
                max_y = y_values.max()
                if max_y > 50:
                    y_step = 15
                elif max_y > 30:
                    y_step = 10
                elif max_y > 15:
                    y_step = 5
                elif max_y > 5:
                    y_step = 2
                else:
                    y_step = 1
                y = range(0, int((max_y//y_step)*y_step+2*y_step), y_step)
                # Identifying the chart position in chart area
                xp = c%4
                yp = c//4
                # Defining chart title and bar color
                if chart_data.columns[c]=="arab_bloc":
                    color = "green"
                    title = "ערביות"
                else:
                    color = party_lists[chart_data.columns[c]]["color"]
                    title = party_lists[chart_data.columns[c]]["title"]
                # Setting chart details:
                # Bars
                ax[yp,xp].bar(x,y_values,width=0.9, facecolor=color)
                # Placement on chart area
                ax[yp,xp].set_position([0.05+0.25*xp,0.68-0.3*yp,0.19,0.175])
                # Ticks
                ax[yp,xp].set_xticks(x)
                ax[yp,xp].set_yticks(y)
                # Tick labels
                ax[yp,xp].set_xticklabels(x+1,fontsize=7)    
                ax[yp,xp].set_yticklabels(y,fontsize=7)
                # Grid lines
                ax[yp,xp].grid(axis='y')
                ax[yp,xp].set_axisbelow(True)
                # Index range separators
                ymx = max_y if max_y > 1 else 1
                ax[yp,xp].vlines(x=[0.45,1.45,5.45,6.45],ymin=0,ymax=ymx,color='gray',linestyle=':',linewidth=0.5)
            # Title
            ax[yp,xp].set_title(bidialg.get_display(title), color=color, pad=2.0)
            # Displaying the figure
            plt.show()
            print ("Graph printed")
        except:
            print("Problem printing Jewish votes by index graph.")

    # Creating graph for potential distinctions between indices
    if create_graphs:
        try:
            # Configuring figure and chart area
            fig, ax = plt.subplots(1,3,figsize=(7,2))
            fig.tight_layout()
            # 1st chart: Right-Haredi in indices 3-5
            x = numpy.array([2,3,4])
            ax[0].bar(x, (jewish_means_by_index["right_bloc"].iloc[2:5] - jewish_means_by_index["haredi_bloc"].iloc[2:5])*100)
            ax[0].set_position([0.05,0.15,0.25,0.7])
            ax[0].set_title(bidialg.get_display("ימין פחות חרדים"))
            ax[0].set_xticks(x)
            ax[0].set_xticklabels(x+1)
            ax[0].grid(axis='y')
            ax[0].set_axisbelow(True)
            # 2nd chart: Left-Haredi in indices 5-6
            x = numpy.array([4,5])
            ax[1].bar(x, (jewish_means_by_index["left_bloc"].iloc[4:6] - jewish_means_by_index["haredi_bloc"].iloc[4:6])*100)
            ax[1].set_position([0.38,0.15,0.25,0.7])
            ax[1].set_title(bidialg.get_display("מרכז-שמאל פחות חרדים"))
            ax[1].set_xticks(x)
            ax[1].set_xticklabels(x+1)
            ax[1].grid(axis='y')
            ax[1].set_axisbelow(True)
            # 3rd chart: Left-Right in indices 8-10
            x = numpy.array([7,8,9])
            ax[2].bar(x, (jewish_means_by_index["left_bloc"].iloc[7:10] - jewish_means_by_index["right_bloc"].iloc[7:10])*100)
            ax[2].set_position([0.71,0.15,0.25,0.7])
            ax[2].set_title(bidialg.get_display("מרכז-שמאל פחות ימין"))
            ax[2].set_xticks(x)
            ax[2].set_xticklabels(x+1)
            ax[2].grid(axis='y')
            ax[2].set_axisbelow(True)
            # Displaying the figure        
            plt.show()
            print("Graph printed")
        except:
            print("Problem printing potential distinctions graph")

        # Creating graph for leftist exceptions
        # Organizing data for leftist kalpiot (60%< center-left votes) with SEI<7
        leftist_exception_data = jewish_clustered_kalpi_data.loc[jewish_clustered_kalpi_data["cluster"]<6].loc[\
            jewish_clustered_kalpi_data["PH"]+jewish_clustered_kalpi_data["KN"]+jewish_clustered_kalpi_data["AMT"]+jewish_clustered_kalpi_data["MRC"]>0.6]
        leftist_exception_data = leftist_exception_data[["place_name","cluster","PH","KN","AMT","MRC","right_bloc","haredi_bloc","L","arab_bloc","others"]]
        leftist_exception_data["right_haredi"] = leftist_exception_data["right_bloc"] + leftist_exception_data["haredi_bloc"]
        leftist_exception_data["others_tmp"] = leftist_exception_data["L"] + leftist_exception_data["arab_bloc"] + leftist_exception_data["others"]
        leftist_exception_means = leftist_exception_data[["PH","KN","AMT","MRC","right_haredi","others_tmp"]].mean()
        leftist_exception_stds = leftist_exception_data[["PH","KN","AMT","MRC","right_haredi","others_tmp"]].std()
        leftist_exception_x_titles = ["פה","כן","אמת","מרצ","ימין-חרדים","אחרות"]
        # Organizing data for the extreme example of Gan Shmuel
        gan_shmuel_data = leftist_exception_data.loc[leftist_exception_data["place_name"]=="GN XMOAL"].copy()
        gan_shmuel_data["all_others"] = gan_shmuel_data["right_haredi"] + gan_shmuel_data["others_tmp"]
        gan_shmuel_means = gan_shmuel_data[["MRC","PH","KN","AMT","all_others"]].mean()
        gan_shmuel_x_titles = ["מרצ","פה","כן","אמת","אחרות"]
        
        leftist_title = "Ballots(%) in " + str(leftist_exception_data.shape[0]) + " Leftist kalpiot with SEI<7"
        gan_shmuel_title = "Extreme example: Gan Shmuel (SEI=3)"

        try:
            # Configuring figure and chart area
            fig, ax = plt.subplots(1,2,figsize=(7,1.5))
            fig.tight_layout()
            # 1st chart: leftist kalpiot
            x = numpy.arange(len(leftist_exception_means))
            y = range(0,50,10)
            # Setting chart details
            # Bars
            ax[0].bar(x, leftist_exception_means*100, yerr=leftist_exception_stds*100)
            # Placement on chart area
            ax[0].set_position([0.05,0.15,0.4,0.7])
            # Ticks
            ax[0].set_xticks(x)
            ax[0].set_yticks(y)
            ax[0].tick_params(axis="x",length=0.0, pad=2.0)
            # Tick labels
            ax[0].set_xticklabels(list(map(lambda s: bidialg.get_display(s), leftist_exception_x_titles)), fontsize=8)
            ax[0].set_yticklabels(y, fontsize=8)
            # Grid lines
            ax[0].grid(axis='y')
            ax[0].set_axisbelow(True)
            # Chart title
            ax[0].set_title(leftist_title, fontsize=9)
            # 2nd chart: Gan Shmuel
            x = numpy.arange(len(gan_shmuel_means))
            # Setting chart details
            # Bars
            ax[1].bar(x, gan_shmuel_means*100)
            # Placement on chart area
            ax[1].set_position([0.55,0.15,0.4,0.7])
            # Ticks
            ax[1].set_xticks(x)
            ax[1].set_yticks(y)
            ax[1].tick_params(axis="x",length=0.0, pad=2.0)
            # Tick labels
            ax[1].set_xticklabels(list(map(lambda s: bidialg.get_display(s), gan_shmuel_x_titles)), fontsize=9)
            ax[1].set_yticklabels(y, fontsize=8)
            # Grid lines
            ax[1].grid(axis='y')
            ax[1].set_axisbelow(True)
            # Chart title
            ax[1].set_title(gan_shmuel_title, fontsize=9)
            # Displaying the figure
            plt.show()
            print("Graph printed")
        except:
            print("Problem printing leftist exception graph")


    # Printing results by index in Arab kalpiot
    arab_clustered_kalpi_data["all_others"] = arab_clustered_kalpi_data["AMT"] + arab_clustered_kalpi_data["B"] + arab_clustered_kalpi_data["G"] +\
                                              arab_clustered_kalpi_data["T"] + arab_clustered_kalpi_data["L"] + arab_clustered_kalpi_data["SS"] +\
                                              arab_clustered_kalpi_data["others"]
    
    arab_means_by_index = arab_clustered_kalpi_data[["cluster","prop_voters","AM","WM","D","MRC","PH","KN","MHL","all_others"]].\
                          groupby(["cluster"]).mean().reset_index().sort_values(by=["cluster"])

    print("\nArab results by index:")
    print("Index","Legal",sep="\t",end="\t")
    for c in list(arab_means_by_index)[2:]:
        title = party_lists[c]["title"] if c in party_lists.keys() else c
        print(title, end="\t")
    print()
        
    for i in range(7):
        print(i+1, (arab_means_by_index.iloc[[i],1:]*100).round(2).to_string(index=False,header=False), sep="\t")
        
    # Creating graph for Arab kalpiot by index
    if create_graphs:
        # Data for the charts
        chart_data = arab_means_by_index[["AM","WM","D","MRC","PH","KN","MHL","all_others"]]
        n_charts = chart_data.shape[1]+1
        try:
            # Configuring figure and chart area
            fig, ax = plt.subplots(3,3,figsize=(3.5,3.5))
            fig.tight_layout()
            fig.suptitle("Party list results (%) by SE-index in Arab sector", fontsize=10)
            # Iterating through charts
            for c in range(n_charts):
                # Obtaining chart title, color and x,y tick values
                if c < n_charts-1:
                    x = numpy.arange(chart_data.shape[0])
                    y_values = chart_data.iloc[:,c].to_numpy()*100
                    show_vlines = True
                    title = party_lists[chart_data.columns[c]]["title"] if chart_data.columns[c] in party_lists.keys() else "אחרות" 
                    color = party_lists[chart_data.columns[c]]["color"] if chart_data.columns[c] in party_lists.keys() else "gray"
                else:
                    x = numpy.array([1,2,3])
                    wmdam = chart_data["WM"] + chart_data["D"] - chart_data["AM"]
                    y_values = wmdam.iloc[1:4].to_numpy()*100
                    show_vlines = False
                    title = "עם - (ד+ום)"
                    color = "teal"
                max_y = y_values.max()
                if max_y > 50:
                    y_step = 15
                elif max_y > 30:
                    y_step = 10
                elif max_y > 15:
                    y_step = 5
                elif max_y > 5:
                    y_step = 2
                else:
                    y_step = 1
                y = range(0, int((max_y//y_step)*y_step+2*y_step), y_step)
                # Identifying chart position in chart area
                xp = c%3
                yp = c//3
                # Setting chart details
                # Bars
                ax[yp,xp].bar(x,y_values,width=0.8, facecolor=color)
                # Placement on chart area
                ax[yp,xp].set_position([0.075+0.325*xp,0.7-0.3*yp,0.225,0.175])
                # Ticks
                ax[yp,xp].set_xticks(x)
                ax[yp,xp].set_yticks(y)
                # Tick labels
                ax[yp,xp].set_xticklabels(x+1,fontsize=7)    
                ax[yp,xp].set_yticklabels(y,fontsize=7)
                # Grid lines
                ax[yp,xp].grid(axis='y')
                ax[yp,xp].set_axisbelow(True)
                # Index range separators
                if show_vlines:
                    ymx = y_values.max() if y_values.max() > 1 else 1
                    ax[yp,xp].vlines(x=[0.45,3.45,4.45],ymin=0,ymax=ymx,color='gray',linestyle=':',linewidth=0.5)
                # Title
                ax[yp,xp].set_title(bidialg.get_display(title), color=color,pad=2.0)
            # Displaying the figure
            plt.show()
            print ("Graph printed")            
        except:
            print("Problem printing Arab votes by index graph")

    
    # Printing results by index in mixed kalpiot
    mixed_clustered_kalpi_data["arab_bloc"] = mixed_clustered_kalpi_data["WM"] + mixed_clustered_kalpi_data["D"] + mixed_clustered_kalpi_data["AM"]
    mixed_clustered_kalpi_data["r_h_bloc"] = mixed_clustered_kalpi_data["MHL"] + mixed_clustered_kalpi_data["T"] + mixed_clustered_kalpi_data["B"] +\
                                             mixed_clustered_kalpi_data["SS"] + mixed_clustered_kalpi_data["G"]
    mixed_clustered_kalpi_data["left_bloc"] = mixed_clustered_kalpi_data["PH"] + mixed_clustered_kalpi_data["KN"] +\
                                               mixed_clustered_kalpi_data["AMT"] + mixed_clustered_kalpi_data["MRC"]
    mixed_means_by_index = mixed_clustered_kalpi_data[["cluster","prop_voters","MHL","T","SS","WM","D","AM","PH","KN","AMT","MRC","L","others",\
                                                        "arab_bloc","r_h_bloc","left_bloc"]].groupby(["cluster"]).mean().reset_index().sort_values(by=["cluster"])
    print("\nMixed locality results by index:")
    print("Index","Legal",sep="\t",end="\t")
    for c in list(mixed_means_by_index)[2:]:
        title = party_lists[c]["title"] if c in party_lists.keys() else c
        print(title, end="\t")
    print()
        
    for i in range(7):
        se_index = int(float(mixed_means_by_index.iloc[[i],0:1].round(0).to_string(index=False,header=False)))
        print(se_index+1, (mixed_means_by_index.iloc[[i],1:]*100).round(2).to_string(index=False,header=False), sep="\t")
    
    # Creating graph for Mixed kalpiot by index
    if create_graphs:
        try:
            chart_data = mixed_means_by_index[["MHL","T","SS","WM","D","AM","PH","KN","AMT","MRC","L","others","arab_bloc","r_h_bloc","left_bloc"]]
            # configuring figure and chart area
            fig, ax = plt.subplots(5,3,figsize=(3.5,4.5))
            fig.tight_layout()
            fig.suptitle("Party results (%) by SE-index in mixed localities", fontsize=10)

            x = numpy.arange(0,6)
            # Iterating through charts
            for c in range(chart_data.shape[1]):
                # Obtaining chart title, color and x,y tick values
                
                if c < 12:
                    title = party_lists[chart_data.columns[c]]["title"]
                    color = party_lists[chart_data.columns[c]]["color"]
                else:
                    if chart_data.columns[c]=="arab_bloc":
                        title = "ערביות"
                        color = "teal"
                    elif chart_data.columns[c]=="r_h_bloc":
                        title = "ימין-חרדים"
                        color = "#50AAFF"
                    else:
                        title = "מרכז-שמאל"
                        color = "#AA9955"
                y_values = chart_data.iloc[0:6,c].to_numpy()*100
                max_y = y_values.max()
                if max_y > 50:
                    y_step = 20
                elif max_y > 25:
                    y_step = 10
                elif max_y > 12:
                    y_step = 5
                elif max_y > 6:
                    y_step = 3
                else:
                    y_step = 1
                y = range(0, int((max_y//y_step)*y_step+2*y_step), y_step)
                # Identifying chart position in chart area
                xp = c%3
                yp = c//3
                # Setting chart details
                # Bars
                ax[yp,xp].bar(x,y_values,width=0.9,facecolor=color)
                # Placement on chart area
                ax[yp,xp].set_position([0.08+0.33*xp,0.8-0.185*yp,0.23,0.11])
                # Ticks
                ax[yp,xp].set_xticks(x)
                ax[yp,xp].set_yticks(y)
                # Tick labels
                ax[yp,xp].set_xticklabels(x+2,fontsize=7)    
                ax[yp,xp].set_yticklabels(y,fontsize=7)
                # Grid lines
                ax[yp,xp].grid(axis='y')
                ax[yp,xp].set_axisbelow(True)
                # Title
                ax[yp,xp].set_title(bidialg.get_display(title),color=color,pad=2.0,fontsize=8)
            # Displaying the figure
            plt.show()
            print ("Graph printed")            
        except:
            print("Problem printing mixed votes by index graph")
     
    # Printing results by index in Druze kalpiot
    druze_clustered_kalpi_data["all_others"] = druze_clustered_kalpi_data["G"] + druze_clustered_kalpi_data["T"] + druze_clustered_kalpi_data["others"]
    druze_means_by_index = druze_clustered_kalpi_data[["cluster","prop_voters","KN","PH","MRC","AMT","L","WM","D","AM","MHL","SS","B","all_others"]].\
                           groupby(["cluster"]).mean().reset_index().sort_values(by=["cluster"])
    
    print("\nDruze results by index:")
    print("Index","Legal",sep="\t",end="\t")
    for c in list(druze_means_by_index)[2:]:
        title = party_lists[c]["title"] if c in party_lists.keys() else c
        print(title, end="\t")
    print()

    for i in range(4):
        se_index = int(float(druze_means_by_index.iloc[[i],0:1].round(0).to_string(index=False,header=False)))
        print(se_index+1, (druze_means_by_index.iloc[[i],1:]*100).round(2).to_string(index=False,header=False), sep="\t")
    
