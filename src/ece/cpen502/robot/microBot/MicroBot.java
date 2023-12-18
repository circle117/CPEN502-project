package ece.cpen502.robot.microBot;

import ece.cpen502.robot.Action;
import ece.cpen502.robot.State;
import robocode.*;

public class MicroBot extends AdvancedRobot {

    protected Action action = new Action();
    protected State state = new State();

    // training parameters
    protected double reward = 0.0;
    protected final double badReward = -0.01;
    protected final double goodReward = 0.01;
    protected final double badTerminalReward = -0.02;
    protected final double goodTerminalReward = 0.02;
    protected final boolean ifTrain = true;

    // robot setting
    protected final EnemyBot enemy = new EnemyBot();
    protected int fireVariable = 400;
    protected byte moveDirection = 1;

    public double normalizeBearing (double angle) {
        while (angle > 180) angle -= 360;
        while (angle < -180) angle += 360;
        return angle;
    }

    @Override
    public void onBulletHit(BulletHitEvent event) {
        if (ifTrain) {
            reward += goodReward;
        }
    }

    @Override
    public void onHitByBullet(HitByBulletEvent event) {
        if (ifTrain) {
            reward += badReward;
        }
    }

    @Override
    public void onHitWall(HitWallEvent event) {
        action.setLabel(Action.ActionLabel.HEAD2CENTER);
        action.doAction(this, null);
    }

    public void setMoveDirection(byte moveDirection) {
        this.moveDirection = moveDirection;
    }

    public byte getMoveDirection() {
        return moveDirection;
    }

    public EnemyBot getEnemy() {
        return enemy;
    }

    public int getFireVariable() {
        return fireVariable;
    }
}
