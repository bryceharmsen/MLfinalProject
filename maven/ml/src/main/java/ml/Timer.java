package ml;

public class Timer {
    
    private double time;
    private double temp;

    public Timer() {
        this.time = -1.0;
        this.temp = 0.0;
    }

    public Timer(Timer copy) {
        this.time = copy.time;
        this.temp = 0.0;
    }

    public void start() throws Exception {
        if (this.temp != 0)
            throw new Exception("Timer was already started.");
        this.temp = System.nanoTime();
    }

    public void end() throws Exception {
        if (this.temp == 0)
            throw new Exception("Timer was already stopped.");
        this.time = System.nanoTime() - this.temp;
        this.temp = 0;
    }

    public double getTime() throws Exception{
        if (this.time < 0)
            throw new Exception("Timer has not executed.");
        return this.time;
    }
}