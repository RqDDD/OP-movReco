import matplotlib.pyplot as plt
from pandas import read_csv
from matplotlib.animation import FuncAnimation
import numpy as np
from mpl_toolkits.mplot3d import axes3d



df = read_csv("/home/rqd/OPlstm/data_csv/d_updown1.csv")

points = df.values

test = df.columns[1:]
liste = points[:,1:]


def display_points(image_x):
    
    ''' Display points on 2D plan / consider there is none aberrant point '''
    
    print(len(test))
    print(len(image_x))
    #listec = listec[1:] # label first
    x, y = [], []
    for a in range(len(image_x)):
        # confidence, x, y we just want x,y
        if a%3 == 1: # x
            print("x", test[a])
            x.append(image_x[a])
            
            
        elif a%3 == 2: # y
            print("y",test[a])
            y.append(image_x[a])
            

    plt.figure('1')
    plt.scatter(x, y, alpha=1)
    plt.show()
    plt.close()

#display_points(liste[85])

def clean_points(listec):
    
    ''' Delete aberrant point to prevent display flaws '''

    # spot aberrant points
    list_ind_aber = []
    for a in range(len(listec)):
        if a%3 == 1: # coord x
            if listec[a] == 0 and listec[a+1] == 0:
                list_ind_aber.append([a,a+1])

    # "correct" aberrant points, assign it other points mean
    nbre_pts = len(listec)/3
    nbr_aber = len(list_ind_aber)
    mean_x = sum(listc[a] )
    mean_y = sum(listc[a] )

    return(listec)


def anim_seq(liste):
    
    ''' Generate a film of points on a 2D plan '''
    
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = ax.plot([], [], 'ro', animated=True)

    def init():
        ax.set_ylim(-1000, 0)
        ax.set_xlim(-1000, 0)
        del xdata[:]
        del ydata[:]
        ln.set_data(xdata, ydata)
        return ln,

    def data_gen(liste):
        data = []
        len_ima = len(liste[0])
        
        for b in range(len(liste)):
            x, y = [], []
            image_x = liste[b]
            for a in range(len_ima):
                # confidence, x, y we just want x,y
                if a%3 == 1: # x    
                    x.append(image_x[a])
                               
                elif a%3 == 2: # y              
                    y.append(image_x[a])
                    
            # add an image        
            data.append([x,y])
        return(data)
        
    def update(frame):

        xdata = frame[0]
        ydata = frame[1]
        ln.set_data(np.negative(xdata), np.negative(ydata))
        return ln,

    ani = FuncAnimation(fig, update, frames = data_gen(liste),
                        init_func=init, blit=True)
    plt.show()


#ani.save('Test.gif', dpi=80, writer='imagemagick')


def mat_rot_y(angle):
    
    ''' rotation matrix about the y-axis. angle in degrees '''

    rangle = angle/360*2*np.pi # angle in radians
    
    mat = np.array([[np.cos(rangle), 0, np.sin(rangle)], [0, 1, 0], [-np.sin(rangle), 0, np.cos(rangle)]])

    return(mat)



def rotate_point(line, angle):
    
    ''' rotate body points around the about the y-axis. angle in degrees, line without label'''

    mat_rot = mat_rot_y(angle)
    line_3d_point = []   # [[nose_x, nose_y, noze_z], [], ...]

    # initialy, we got a 2D plan, z coordonate = 1

    for a in range(len(line)-1):
        if a%3 == 1: # x    
            line_3d_point.append(np.dot(mat_rot, np.array([line[a], line[a+1], 1])))

    return(line_3d_point)

# rotate_point(liste[0], 10)

def anim_rot_3d(line):
    
    ''' Generate a film of points in 3D illustrating rotation. Visual testing of rotation
    line without label  '''


    X = []
    Y = []
    Z = []
    xdata, ydata, zdata = [], [], []
    
    theta = np.linspace(2,100,10)    
    points_3d = rotate_point(line, 10)
    def data_gen(theta, line):
        data = []
        X = []
        Y = []
        Z = []
        for b in range(len(theta)):
            points_3d = rotate_point(line, theta[b])
            for a in range(len(points_3d)):
                X.append(-points_3d[a][0])
                Y.append(-points_3d[a][2])
                Z.append(-points_3d[a][1])
            data.append([X,Y,Z])
        
        
        #print(type(data))
        
        return(np.array(data))
    #print(data_gen(theta))
    
##    def update_lines(num, dataLines, lines):
##        print(type(dataLines))
##        lines[0].set_data(dataLines[num,0:2])
##        lines[0].set_3d_properties(dataLines[num,2])
##        print(lines[0])  
##        return lines[0]

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = axes3d.Axes3D(fig)

    data = data_gen(theta, line)
    print(data)
    #print(data)
    #print(type(data[0]))
    for a in range(len(data)):       
        lines = ax.plot(data[a, 0], data[a, 1], data[a, 2], 'ro')[0]


    
##    line_ani = FuncAnimation(fig, update_lines, 25, fargs=(data, lines),
##                                   interval=200, blit=False)

    plt.show()



##    X = []
##    Y = []
##    Z = []
##    
##
##    points_3d = rotate_point(line, 0)
##    
##    for a in range(len(points_3d)):
##        X.append(-points_3d[a][0])
##        Y.append(-points_3d[a][2])
##        Z.append(-points_3d[a][1]) # just visual


##    fig = plt.figure()
##    ax = fig.add_subplot(111, projection='3d')
##
##    ax.scatter(X, Y, Z)
##    plt.show()

        
#anim_rot_3d(liste[0])    
    
