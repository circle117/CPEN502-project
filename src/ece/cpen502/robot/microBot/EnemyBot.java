package ece.cpen502.robot.microBot;

import robocode.ScannedRobotEvent;

public class EnemyBot {
    private double bearing;
    private double distance;
    private double energy;
    private double heading;
    private String name;
    private double velocity;

    public EnemyBot(){
        reset();
    }

    public double getBearing() {
        return bearing;
    }

    public double getDistance() {
        return distance;
    }

    public double getEnergy() {
        return energy;
    }

    public double getHeading() {
        return heading;
    }

    public String getName() {
        return name;
    }

    public double getVelocity() {
        return velocity;
    }

    public void update(ScannedRobotEvent event) {
        bearing = event.getBearing();
        distance = event.getDistance();
        energy = event.getEnergy();
        heading = event.getHeading();
        name = event.getName();
        velocity = event.getVelocity();
    }

    public void reset() {
        bearing = 0.0;
        distance = 0.0;
        energy = 0.0;
        heading = 0.0;
        name = "";
        velocity = 0.0;
    }

    public boolean none() {
        if (name.equals(""))
            return true;
        else
            return false;
    }
}
